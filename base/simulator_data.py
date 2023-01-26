import logging
from typing import Optional, List, Union

import pandas as pd

from base.window_generator import WindowGenerator
from common import preprocess_df, split_df, normalize_df


class SimulatorData:
    """Used for accessing initial and continual data for simulations."""

    def __init__(self,
                 initial_source: str,
                 initial_start: str,
                 initial_end: str,
                 continual_start: str,
                 continual_end: str,
                 continual_source: Optional[str] = None
                 ) -> None:
        self.initial_source = initial_source
        self.initial_start = initial_start
        self.initial_end = initial_end
        self.continual_start = continual_start
        self.continual_end = continual_end
        self.continual_source = continual_source

        if self.continual_start <= self.initial_end:
            logging.warning('Continual data is overlapping with initial data.')

        # lazy loading
        self._initial_df = None
        self._continual_df = None

    @property
    def initial_df(self) -> pd.DataFrame:
        if self._initial_df is None:
            self._load_data()
        return self._initial_df.copy()

    @property
    def continual_df(self) -> pd.DataFrame:
        if self._continual_df is None:
            self._load_data()
        return self._continual_df.copy()

    def get_window_generator(self,
                             input_features: List[str],
                             output_features: List[str],
                             periodicity: List[str],
                             input_length: int,
                             output_length: int,
                             stride: int,
                             sampling_rate: int,
                             batch_size: int = 32,
                             validation_split: Optional[Union[str, float]] = 0.8,
                             exclude_normalization: Optional[List[str]] = None,
                             ) -> WindowGenerator:
        """Returns a window generator containing the preprocessed initial and continual data.

        The continual data will be the test data of the window generator.
        The initial data will be split in train and validation data using the validation_split parameter.
        If validation_split is a float, it will be used as the percentage of the initial data that will be
        used for validation.
        If validation_split is a string, it will be used as the period of the initial data, e.g. '4w' for the
        last 4 weeks.
        If validation_split is None, there will be no validation data.
        """
        initial_df = preprocess_df(self.initial_df, input_features, periodicity)
        continual_df = preprocess_df(self.continual_df, input_features, periodicity)

        if validation_split is None:
            train_df, val_df = initial_df.copy(), None
        elif isinstance(validation_split, float):
            train_df, val_df, _ = split_df(initial_df, 1 - validation_split, validation_split)
        else:
            validation = pd.Timedelta(validation_split)
            train_df = initial_df.loc[:initial_df.index[-1] - validation, :].copy()
            val_df = initial_df.loc[initial_df.index[-1] - validation:, :].copy()

        norm_mean, norm_std = normalize_df(input_features, train_df, [val_df, continual_df], exclude_normalization)
        return WindowGenerator(input_length, output_length, stride, sampling_rate,
                               train_df, val_df, continual_df,
                               norm_mean, norm_std, periodicity,
                               input_features, output_features, batch_size
                               )

    def _load_data(self):
        df = pd.read_pickle(self.initial_source)
        self._initial_df = df.loc[self.initial_start:self.initial_end].copy()
        if self.continual_source is not None:
            df = pd.read_pickle(self.continual_source)
        self._continual_df = df.loc[self.continual_start:self.continual_end].copy()

    def to_dict(self) -> dict:
        return {
            'initial_source': self.initial_source,
            'initial_start': self.initial_start,
            'initial_end': self.initial_end,
            'continual_start': self.continual_start,
            'continual_end': self.continual_end,
            'continual_source': self.continual_source,
        }

    @staticmethod
    def from_dict(d: dict) -> 'SimulatorData':
        return SimulatorData(**d)
