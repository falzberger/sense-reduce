import datetime
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import full_hours_before, to_full_hour


# TODO (long-term): this class should become a layer for accessing a time-series database (e.g., InfluxDB)
class DataStorage:
    """A wrapper around two pandas Dataframe for past measurements and predictions."""

    def __init__(self, input_features: List[str], output_features: List[str]) -> None:
        self._measurements = pd.DataFrame(columns=input_features, dtype=np.float64)
        self._predictions = pd.DataFrame(columns=output_features, dtype=np.float64)

    @property
    def mae(self) -> pd.Series:
        return self.get_diff().abs().mean()

    @property
    def mse(self) -> pd.Series:
        return (self.get_diff() ** 2).mean()

    @property
    def rmse(self) -> pd.Series:
        return self.mse ** 0.5

    @classmethod
    def from_data(cls, measurements: pd.DataFrame, predictions: pd.DataFrame) -> 'DataStorage':
        storage = cls(measurements.columns, predictions.columns)
        storage._measurements = measurements
        storage._predictions = predictions
        return storage

    @classmethod
    def from_previous_years_average(cls, start: datetime.datetime,
                                    end: datetime.datetime,
                                    previous: pd.DataFrame,
                                    output_features: List[str]) -> 'DataStorage':
        """Creates StorageData with measurements set in the given range using average previous values."""
        # TODO: add a parameter max_prev_years to limit the number of previous years to use
        assert end > start, "Expected end date to be after start date"
        data = DataStorage(previous.columns, output_features)
        i = to_full_hour(start)  # force full-hours
        while i <= end:
            j = i - datetime.timedelta(days=365)  # this way we don't have to consider Feb 29 separately
            previous_values = pd.DataFrame(columns=previous.columns)
            value = previous.loc[j]
            while value is not None:
                previous_values.loc[j] = value
                j -= datetime.timedelta(days=365)
                try:
                    value = previous.loc[j]
                except KeyError:
                    value = None
            data.add_measurement(i, previous_values.mean())
            i += datetime.timedelta(hours=1)
        return data

    @classmethod
    def csv_path_measurements(cls, prefix='') -> str:
        if prefix != '':
            return f'{prefix}_measurements.csv'
        return f'measurements.csv'

    @classmethod
    def csv_path_predictions(cls, prefix='') -> str:
        if prefix != '':
            return f'{prefix}_predictions.csv'
        return f'predictions.csv'

    def add_measurement(self, dt: datetime.datetime, values: np.ndarray):
        self._measurements.loc[dt] = values

    def add_prediction(self, dt: datetime.datetime, values: np.ndarray):
        self._predictions.loc[dt] = values

    def add_measurement_dict(self, d: dict):
        for date_string, values in d.items():
            self.add_measurement(datetime.datetime.fromisoformat(date_string), values)

    def add_prediction_dict(self, d: dict):
        for date_string, values in d.items():
            self.add_prediction(datetime.datetime.fromisoformat(date_string), values)

    def add_measurement_df(self, df: pd.DataFrame):
        self._measurements = pd.concat([self._measurements, df], copy=False)

    def add_prediction_df(self, df: pd.DataFrame):
        self._predictions = pd.concat([self._predictions, df], copy=False)

    def copy(self, deep=True) -> 'DataStorage':
        """Returns a deep copy of this object. Note that the csv_path is also equal unless changed afterwards."""
        copy = DataStorage(self._measurements.columns, self._predictions.columns)
        copy._measurements = self._measurements.copy(deep)
        copy._predictions = self._predictions.copy(deep)
        return copy

    def get_measurements(self) -> pd.DataFrame:
        return self._measurements

    def get_predictions(self) -> pd.DataFrame:
        return self._predictions

    def get_measurements_previous_hours(self, dt: datetime.datetime, n_hours: int) -> pd.DataFrame:
        """Returns the measurements at the full hours before the specified timestamp (inclusive).

        If there are no measurements for a full hour, the values of the next one are used.
        """
        hours = list(full_hours_before(dt, n_hours))  # will result in an already sorted list
        idx = self._measurements.index.get_indexer(hours, method='nearest')
        result: pd.DataFrame = self._measurements.iloc[idx].copy()
        result.set_index(pd.DatetimeIndex(hours), inplace=True)
        return result

    def get_diff(self, columns: List[str] = None) -> pd.DataFrame:
        """Returns the difference between measurements and predictions. Removes NaNs."""
        diff: pd.DataFrame = self._measurements.loc[:, self._predictions.columns] - self._predictions
        if columns is not None:
            diff = diff[columns]
        return diff[~diff.isna().any(axis=1)]

    def plot(self):
        """Creates a plot for every attribute, comparing measurements and predictions."""
        for col in self._measurements.columns:
            plt.plot(self._measurements[col], label='Measurement')
            plt.plot(self._predictions[col], label='Prediction')
            plt.show()

    def save(self, dir_path='.', csv_prefix='') -> None:
        os.makedirs(dir_path, exist_ok=True)
        self._measurements.to_csv(os.path.join(dir_path, self.csv_path_measurements(csv_prefix)), index=True)
        self._predictions.to_csv(os.path.join(dir_path, self.csv_path_predictions(csv_prefix)), index=True)

    @classmethod
    def load(cls, dir_path='.', csv_prefix='') -> 'DataStorage':
        measurements = pd.read_csv(os.path.join(dir_path, DataStorage.csv_path_measurements(csv_prefix)),
                                   index_col=0, parse_dates=True)
        predictions = pd.read_csv(os.path.join(dir_path, DataStorage.csv_path_predictions(csv_prefix)),
                                  index_col=0, parse_dates=True)

        ds = DataStorage(measurements.columns, predictions.columns)
        ds._measurements = measurements
        ds._predictions = predictions
        return ds
