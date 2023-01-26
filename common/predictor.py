import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .data_storage import DataStorage
from .prediction_model import PredictionModel


class PredictionHorizon:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self._tmp = df.copy()

    def __str__(self) -> str:
        return self.df.__str__()

    @property
    def start(self) -> datetime:
        return self.df.index.min().to_pydatetime()

    @property
    def end(self) -> datetime:
        return self.df.index.max().to_pydatetime()

    def in_prediction_horizon(self, dt: datetime) -> bool:
        """Whether the provided datetime is within the prediction horizon."""
        return self.start <= dt <= self.end

    def get_prediction_at(self, dt: datetime) -> pd.Series:
        """Returns the interpolated prediction for the specified datetime or a ValueError if out of range.
        The pandas.Series is indexed by the output features."""
        if not self.in_prediction_horizon(dt):
            raise ValueError(f'Datetime {dt} is out of range of prediction horizon {self.start} - {self.end}')

        try:
            return self._tmp.loc[dt]
        except KeyError:
            self._tmp.loc[dt] = np.nan
            self._tmp.interpolate(method='time', inplace=True)
            return self._tmp.loc[dt]


class Predictor:
    """Uses a PredictionModel to provide predictions for arbitrary timestamps in the prediction range of the model."""

    def __init__(self, model: PredictionModel, data: DataStorage) -> None:
        assert model.metadata.output_length <= 24  # self.get_prediction_at(dt) expects a horizon of less than a day
        self._model = model
        self._data = data
        self._prediction_horizon: Optional[PredictionHorizon] = None

    @property
    def model_metadata(self):
        return self._model.metadata

    @property
    def data(self) -> DataStorage:
        return self._data

    @property
    def prediction_horizon_start(self):
        return self._prediction_horizon.start

    @property
    def prediction_horizon_end(self):
        return self._prediction_horizon.end

    def set_model(self, other: PredictionModel, start: datetime):
        """Changes the underlying model of the predictor and resets the prediction horizon to the specified datetime."""
        logging.debug(f'Changing predictor model from "{self._model.metadata.uuid}" to "{other.metadata.uuid}"')
        self._model = other
        self._prediction_horizon = None
        self.update_prediction_horizon(start)

    def add_measurement(self, dt: datetime, values: np.ndarray):
        self._data.add_measurement(dt, values)

    def add_measurement_df(self, df: pd.DataFrame):
        self._data.add_measurement_df(df)

    def add_prediction(self, dt: datetime, values: np.ndarray):
        self._data.add_prediction(dt, values)

    def add_prediction_df(self, df: pd.DataFrame):
        self._data.add_prediction_df(df)

    def get_measurements_in_current_prediction_horizon(self, until: datetime) -> Optional[pd.DataFrame]:
        """Returns the hourly measurements in the current horizon until the specified datetime (inclusive)."""
        if self._prediction_horizon is None:
            return None
        else:
            assert self.in_prediction_horizon(until)
            elapsed_hours = int((until - self.prediction_horizon_start).total_seconds() / 3600)
            return self._data.get_measurements_previous_hours(until, elapsed_hours)

    def get_predictions_until(self, until: datetime) -> Optional[pd.DataFrame]:
        if self._prediction_horizon is None:
            return None
        else:
            # the first interpolation point is a measurement, not a prediction
            return self._prediction_horizon.df.between_time(self.prediction_horizon_start.time(), until.time(),
                                                            inclusive='right'
                                                            )

    def get_prediction_at(self, dt: datetime) -> pd.Series:
        """Returns the interpolated prediction for the specified datetime or None if it is out of range.
        The pandas.Series is indexed by the output features."""
        return self._prediction_horizon.get_prediction_at(dt)

    def in_prediction_horizon(self, dt: datetime) -> bool:
        return self._prediction_horizon.in_prediction_horizon(dt)

    def adjust_to_measurement(self, dt: datetime, measurement: np.ndarray, prediction: np.ndarray) -> None:
        """Makes the predictor aware of a threshold violation so that it can adjust future predictions."""
        if self._prediction_horizon is None:
            logging.error(f'Cannot adapt non-existing interpolation points!')
            return
        diff = measurement[self.model_metadata.input_to_output_indices] - prediction
        logging.debug(f'Adapting prediction at {dt} by {diff}')
        self._prediction_horizon = PredictionHorizon(self._prediction_horizon.df + diff)
        logging.debug(f'New prediction horizon: {self._prediction_horizon}')

    def update_prediction_horizon(self, start: datetime):
        """Updates the interpolation points used for computing predictions. """
        if self._prediction_horizon is not None and \
                self._prediction_horizon.df.index[0] <= start < self._prediction_horizon.df.index[1]:
            logging.debug(f'Skipped updating prediction horizon because nothing would change: '
                          f'{self._prediction_horizon.df.index[0]} <= {start} < {self._prediction_horizon.df.index[1]}'
                          )
            return

        previous_m = self._data.get_measurements_previous_hours(start, self._model.metadata.input_length)
        new_horizon = self._model.predict(previous_m)

        # we also need the last measurement for interpolation
        last_ts = previous_m.index.max()
        new_horizon.loc[last_ts] = previous_m.loc[last_ts, self.model_metadata.output_features]
        new_horizon.sort_index(inplace=True)

        previous_ip = self._prediction_horizon
        self._prediction_horizon = PredictionHorizon(new_horizon)
        logging.debug(f'Updated prediction horizon from\n{previous_ip}\nto\n{self._prediction_horizon}')
