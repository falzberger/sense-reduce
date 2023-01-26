from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


class WindowGenerator:
    """ A helper class for data windowing with time-series data.

    As introduced in the TensorFlow tutorial for time-series prediction
    (cf. https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing).
    """

    def __init__(self, input_length: int, output_length: int, stride: int, sampling_rate: int,
                 train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame],
                 norm_mean: List[float], norm_std: List[float], periodicity: List[str],
                 input_features: List[str], output_features: List[str] = None,
                 batch_size=32, shuffle_batches=True):
        """

        The data frames are expected to be normalized and have the same columns.

        Args:
            input_length:
            output_length:
            stride:
            sampling_rate:
            train_df:
            val_df:
            test_df:
            norm_mean:
            norm_std:
            periodicity:
            input_features:
            output_features:
            batch_size:
            shuffle_batches:
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.periodicity = periodicity
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches

        self.output_features = output_features
        if output_features is not None:
            self.output_features_indices = {name: i for i, name in enumerate(output_features)}
        self.input_features = input_features
        self.input_features_indices = {name: i for i, name in enumerate(self.input_features)}

        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        self.sampling_rate = sampling_rate

        self.total_window_size = input_length + output_length

        self.input_slice = slice(0, input_length)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.output_start = input_length
        self.outputs_slice = slice(self.output_start, None)
        self.output_indices = np.arange(self.total_window_size)[self.outputs_slice]

        self._example = None

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Output indices: {self.output_indices}',
            f'Output column name(s): {self.output_features}'])

    @property
    def input_shape(self):
        return self.input_length, len(self.input_features) + 2 * len(self.periodicity)

    @property
    def output_shape(self):
        if self.output_features is None:
            # encoded periodicity will not be in default output features
            return self.output_length, len(self.input_features)
        return self.output_length, len(self.output_features)

    @property
    def df(self) -> pd.DataFrame:
        """ Returns all data in the window in a single dataframe."""
        return pd.concat([self.train_df, self.val_df, self.test_df])

    @property
    def train(self) -> tf.data.Dataset:
        return self.make_dataset(self.train_df)

    @property
    def val(self) -> Optional[tf.data.Dataset]:
        if self.val_df is None:
            return None
        return self.make_dataset(self.val_df)

    @property
    def test(self) -> Optional[tf.data.Dataset]:
        if self.test_df is None:
            return None
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, outputs` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        outputs = features[:, self.outputs_slice, :]
        if self.output_features is not None:
            outputs = tf.stack(
                [outputs[:, :, self.input_features_indices[name]] for name in self.output_features],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_length, None])
        outputs.set_shape([None, self.output_length, None])

        return inputs, outputs

    def make_dataset(self, data) -> tf.data.Dataset:
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.stride,
            sampling_rate=self.sampling_rate,
            shuffle=self.shuffle_batches,
            batch_size=self.batch_size,
        )

        return ds.map(self.split_window)

    def plot(self, plot_col: str, model=None, max_subplots=3, title=None):
        inputs, outputs = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.input_features_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.output_features:
                output_col_index = self.output_features_indices.get(plot_col, None)
            else:
                output_col_index = plot_col_index

            if output_col_index is None:
                continue

            plt.scatter(self.output_indices, outputs[n, :, output_col_index],
                        edgecolors='k', label='Outputs', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.output_indices, predictions[n, :, output_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        if title:
            plt.suptitle(title)

        plt.xlabel('Time [h]')
        plt.show()

    def plot_distribution(self, columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.df.columns
        features = self.df[columns]
        df = features.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df)
        _ = ax.set_xticklabels(features.keys(), rotation=90)
        plt.show()
