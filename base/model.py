import logging
import os
from datetime import datetime
from typing import Any, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from common import full_hours_after, LiteModel, convert_datetime
from common.model_metadata import ModelMetadata
from common.prediction_model import PredictionModel


class Model(PredictionModel):
    """Wrapper for a keras Sequential model and its associated metadata."""

    def __init__(self,
                 model: Union[str, tf.keras.Model],
                 metadata: ModelMetadata
                 ) -> None:
        logging.debug(f'Created new model with metadata {metadata}')
        self._metadata: ModelMetadata = metadata

        if isinstance(model, str):  # lazy loading
            self._model_path = model
            self._model = None
        else:
            self._model_path = None
            self._model: tf.keras.Model = model
            self._check_model()

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            logging.debug(f'Loading model from {self._model_path}')
            self._model = tf.keras.models.load_model(self._model_path)
            self._check_model()
        return self._model

    @property
    def metadata(self) -> ModelMetadata:
        """Returns a reference to the model's metadata. Use deepcopy() if needed."""
        return self._metadata

    @classmethod
    def load(cls, path: str, lazy_loading=False) -> 'Model':
        """Loads the model from the specified directory, assuming it contains a 'metadata.json' file."""
        metadata = ModelMetadata.load(os.path.join(path, ModelMetadata.FILE_NAME))
        if lazy_loading:
            return cls(path, metadata)
        else:
            return cls(tf.keras.models.load_model(path), metadata)

    def _check_model(self):
        if self.model.input_shape != self.metadata.input_shape and self.model.input_shape[0] is not None:
            logging.warning(f'Model input shape and metadata input shape do not match, '
                            f'{self.model.input_shape} != {self.metadata.input_shape}')
        if self.model.output_shape != self.metadata.output_shape and self.model.output_shape[0] is not None:
            logging.warning(f'Model output shape and metadata output shape do not match, '
                            f'{self.model.output_shape} != {self.metadata.output_shape}')

    def _to_tflite_model(self) -> Any:
        """Creates a TFLite model from this model."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        return converter.convert()

    def to_lite_model(self) -> LiteModel:
        """Creates a LiteModel object from this model."""
        interpreter = tf.lite.Interpreter(model_content=self._to_tflite_model())
        interpreter.allocate_tensors()
        return LiteModel(interpreter, self.metadata.deepcopy())

    def clone(self):
        """Creates an identical copy of this model. Note that it must be compiled before use."""
        clone = tf.keras.models.clone_model(self.model)
        clone.set_weights(self.model.get_weights())
        return Model(clone, self.metadata.deepcopy())

    def save(self, model_dir='.') -> None:
        """Saves the model in SavedModel format and its metadata in JSON format."""
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(model_dir)
        self.metadata.save(os.path.join(model_dir, ModelMetadata.FILE_NAME))

    def save_lite(self, model_dir='.'):
        """Saves the model as a TFLite model."""
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, LiteModel.FILE_NAME), 'wb') as f:
            f.write(self._to_tflite_model())
continual_strategy
    def save_and_convert(self, model_dir='.') -> None:
        """Saves the model in SavedModel format and as a TFLite model."""
        self.save(model_dir)
        self.save_lite(model_dir)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Runs a single inference of the model.

        Args:
            x: A pandas.DataFrame with DatetimeIndex

        Returns:
            A pandas DataFrame with DatetimeIndex
        """
        convert_datetime(x, self.metadata.periodicity)
        x_np = self._normalize_input_features(x).values.reshape((1, self.metadata.input_length, -1))

        return self._model_output_to_dataframe(self.model.predict(x_np), x.index.max())

    def _normalize_input_features(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.metadata.input_features] = (x[self.metadata.input_features] - self.metadata.input_normalization_mean) \
                                          / self.metadata.input_normalization_std
        return x

    def _model_output_to_dataframe(self, a: np.ndarray, dt: datetime) -> pd.DataFrame:
        a = (a * self.metadata.output_normalization_std) + self.metadata.output_normalization_mean
        return pd.DataFrame(data=a.reshape((self.metadata.output_length, self.metadata.output_attributes)),
                            columns=self.metadata.output_features,
                            index=full_hours_after(dt, self.metadata.output_length))

    def get_parameter_count(self) -> int:
        return self.model.count_params()

    def get_lite_model_size(self) -> int:
        """Returns the size of the TFLite model in bytes."""
        return len(self._to_tflite_model())
