from abc import ABC, abstractmethod

import pandas as pd

from common.model_metadata import ModelMetadata


class PredictionModel(ABC):
    """A generic interface definition for implementations of time series prediction models."""

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        pass

    @abstractmethod
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        pass
