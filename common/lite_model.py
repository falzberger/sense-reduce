import logging
import os
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .model_metadata import ModelMetadata
from .prediction_model import PredictionModel
from .utils import full_hours_after, convert_datetime


class LiteModel(PredictionModel):
    """Wraps a TensorFlow Lite Interpreter for inference."""

    FILE_NAME = 'model.tflite'

    def __init__(self,
                 interpreter: Union[str, tf.lite.Interpreter],
                 metadata: ModelMetadata
                 ) -> None:
        self._metadata = metadata

        if isinstance(interpreter, str):  # lazy loading
            self._interpreter_path = interpreter
            self._interpreter = None
        else:
            self._interpreter_path = None
            self._interpreter: tf.lite.Interpreter = interpreter
            self._check_interpreter()

    @property
    def interpreter(self) -> tf.lite.Interpreter:
        if self._interpreter is None:
            self._interpreter = self._load_interpreter(self._interpreter_path)
            self._check_interpreter()
        return self._interpreter

    @property
    def metadata(self) -> ModelMetadata:
        """Returns a reference to the interpreter's metadata. Use deepcopy() if needed."""
        return self._metadata

    @classmethod
    def from_tflite_file(cls, path: str, metadata: ModelMetadata) -> 'LiteModel':
        return cls(cls._load_interpreter(path), metadata)

    @classmethod
    def load(cls, path: str, lazy_loading=False) -> 'LiteModel':
        """Loads a model from a directory, assuming a "model.tflite" and "metadata.json" file are present."""
        metadata = ModelMetadata.load(os.path.join(path, ModelMetadata.FILE_NAME))
        if lazy_loading:
            return cls(os.path.join(path, cls.FILE_NAME), metadata)
        else:
            return cls.from_tflite_file(os.path.join(path, cls.FILE_NAME), metadata)

    @staticmethod
    def _load_interpreter(path: str) -> tf.lite.Interpreter:
        logging.info(f'Loading TFLite model from path "{path}"')
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        return interpreter

    def _check_interpreter(self) -> None:
        input_shape = tuple(self.interpreter.get_input_details()[0]['shape'])
        if input_shape != self.metadata.input_shape:
            logging.warning(f'Interpreter input shape and metadata input shape do not match, '
                            f'{input_shape} != {self.metadata.input_shape}')
        output_shape = tuple(self.interpreter.get_output_details()[0]['shape'])
        if output_shape != self.metadata.output_shape:
            logging.warning(f'Interpreter output shape and metadata output shape do not match, '
                            f'{output_shape} != {self.metadata.output_shape}')

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Runs a single inference on the interpreter.

        Args:
            input_df: a pandas DataFrame with DatetimeIndex

        Returns:
            A pandas DataFrame with DatetimeIndex
        """
        last_ts = input_df.index.max()
        input_df = (input_df - self.metadata.input_normalization_mean) / self.metadata.input_normalization_std
        convert_datetime(input_df, self.metadata.periodicity)

        # TFLite interpreter invocation
        input_data = np.array(input_df, dtype=np.float32).reshape(self.metadata.input_shape)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])

        predictions = (predictions * self.metadata.output_normalization_std) + self.metadata.output_normalization_mean
        return pd.DataFrame(data=predictions.reshape((self.metadata.output_length, self.metadata.output_attributes)),
                            columns=self.metadata.output_features,
                            index=full_hours_after(last_ts, self.metadata.output_length))
