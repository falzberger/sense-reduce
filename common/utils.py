import datetime
import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd


def to_full_hour(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour)


def full_hours_before(dt: datetime.datetime, n_hours: int):
    """Yields datetime objects for the last n_hours full hours before dt (inclusive).

    Iterator starts with the (n_hours - 1) before dt and ends with the full hour of dt.
    """
    # dt -= datetime.timedelta(microseconds=1)  # would exclude dt
    last_hour = to_full_hour(dt)
    for i in reversed(range(0, n_hours)):
        yield last_hour - datetime.timedelta(hours=i)


def full_hours_after(dt: datetime.datetime, n_hours: int):
    """Yields datetime objects for the next n_hours full hours after dt (exclusive)."""
    last_hour = to_full_hour(dt)
    for i in range(1, n_hours + 1):
        yield last_hour + datetime.timedelta(hours=i)


def timeseries_to_dict(series) -> dict:
    """Converts a pandas Series with DatetimeIndex to a dictionary where the keys are the timestamps (ISO format)."""
    result = {}
    for index, elem in series.items():
        result[index.to_pydatetime().isoformat()] = elem
    return result


def dict_to_timeseries(d: dict, dtype) -> pd.Series:
    """Converts a dictionary where the keys are timestamps in ISO format to a pandas Series with DatetimeIndex."""
    result = pd.Series(dtype=dtype)
    for index, elem in d.items():
        result[datetime.datetime.fromisoformat(index)] = elem
    return result


def preprocess_df(df: pd.DataFrame, features: List[str], periodicity: List[str]) -> pd.DataFrame:
    """Removes non-feature columns from the DataFrame and encodes the datetime index as periodic functions."""
    df = df.reindex(columns=features, copy=False)
    convert_datetime(df, periodicity)
    return df


def convert_datetime(df: pd.DataFrame, periodicity: List[str]):
    """Adds periodicity to the dataframe by encoding them in sine and cosine function values.

    For more information see
    https://www.tensorflow.org/tutorials/structured_data/time_series#time.
    """
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    periodicity = list(map(lambda x: x.lower(), periodicity))
    normalization_factor = 0.70710651  # = std(sin()) = std(cos())

    if 'day' in periodicity:
        day = 86400  # 24 * 60 * 60
        df.loc[:, 'Day sin'] = np.sin(timestamp_s * (2 * np.pi / day)) / normalization_factor
        df.loc[:, 'Day cos'] = np.cos(timestamp_s * (2 * np.pi / day)) / normalization_factor
    if 'week' in periodicity:
        week = 604800  # 7 * day
        df.loc[:, 'Week sin'] = np.sin(timestamp_s * (2 * np.pi / week)) / normalization_factor
        df.loc[:, 'Week cos'] = np.cos(timestamp_s * (2 * np.pi / week)) / normalization_factor
    if 'year' in periodicity:
        year = 31556952  # 365.2425 * day
        df.loc[:, 'Year sin'] = np.sin(timestamp_s * (2 * np.pi / year)) / normalization_factor
        df.loc[:, 'Year cos'] = np.cos(timestamp_s * (2 * np.pi / year)) / normalization_factor


def normalize_df(features: List[str],
                 train_df: pd.DataFrame,
                 other_dfs: List[Optional[pd.DataFrame]],
                 exclude: Optional[List[str]] = None,
                 ) -> (List[np.float32], List[np.float32]):
    means, stds = [], []
    for col in features:
        if exclude is not None and col in exclude:
            mean, std = 0, 1
        else:
            mean, std = train_df[col].mean(), train_df[col].std()
        means.append(mean)
        stds.append(std)

        train_df.loc[:, col] = (train_df.loc[:, col] - mean) / std
        for df in filter(lambda d: d is not None, other_dfs):
            df.loc[:, col] = (df.loc[:, col] - mean) / std

    logging.debug(f'Normalized features {features}')
    logging.debug(f'Means: {means}')
    logging.debug(f'Stds: {stds}')

    return means, stds


def split_df(df: pd.DataFrame, first_split: float, second_split: float
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataframe into three parts with the specified ratios. Copies the dataframe."""
    assert first_split + second_split <= 1

    n = len(df)
    first_df = df.iloc[0:int(n * first_split)].copy()
    second_df = df.iloc[int(n * first_split):int(n * (first_split + second_split))].copy()
    third_df = df.iloc[int(n * (first_split + second_split)):].copy()

    return first_df, second_df, third_df


def progress_bar(df: pd.DataFrame,
                 prefix=None,
                 suffix='',
                 decimals=2,
                 length=50,
                 fill='=',
                 print_end="\r",
                 verbose: int = 1
                 ):
    """
    Displays a progress bar in the console while iterating over the dataframe. Uses df.itertuples() underneath.
    (similar to https://stackoverflow.com/a/34325723/9548194)

    Parameters:
        df: The pandas DataFrame to iterate over
        prefix: The prefix to display before the progress bar
        suffix: The suffix to display after the progress bar
        decimals: The number of decimals to display
        length: The (char) length of the progress bar
        fill: The character to use for the progress bar
        print_end: The end character (e.g. "\r", "\r\n")
        verbose: Toggles whether the progress bar is updated for every item or not.
    """
    start = time.time()
    total = len(df)
    if prefix is None:
        prefix = f'Iterating over {total} elements'

    def print_progress_bar(iteration):
        percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + ' ' * (length - filled_length)
        now = time.time()
        elapsed = int(now - start)
        print(f'\r{prefix} [{bar}] {iteration}/{total} {percent}% ({elapsed}s) {suffix}', end=print_end)

    print_progress_bar(0)
    for i, item in enumerate(df.itertuples()):
        yield item
        if verbose == 1:
            print_progress_bar(i + 1)
    print_progress_bar(total)
    print()
