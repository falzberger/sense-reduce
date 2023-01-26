import json

from typing import List, Optional

import pandas as pd


class ModelMetadata:
    """Wraps metadata for Machine Learning models, including values for input normalization."""

    FILE_NAME = 'metadata.json'

    def __init__(self,
                 uuid: str,
                 input_features: List[str],
                 input_shape: tuple,
                 output_shape: tuple,
                 periodicity: List[str],
                 normalization_mean: List[float],
                 normalization_std: List[float],
                 output_features: Optional[List[str]] = None,
                 context: Optional[dict] = None
                 ) -> None:
        """input_shape and output_shape are expected to be 3-tuples similar as calling np.ndarray.shape."""
        assert len(normalization_mean) == len(normalization_std) == len(input_features), \
            'Normalization values do not match input features'
        assert len(input_features) + 2 * len(periodicity) == input_shape[2], \
            'Input shape does not match input features.'
        assert len(output_features) == output_shape[2], \
            'Output shape does not match output features'
        assert all(map(lambda f: f in input_features, output_features)), \
            'Output features must be a subset of input features'

        self.uuid = uuid
        self.input_features = input_features
        if output_features is None:
            self.output_features = input_features.copy()
            self.input_to_output_indices = list(range(len(input_features)))
        else:
            self.output_features = output_features
            self.input_to_output_indices = [input_features.index(f) for f in output_features]

        self.periodicity = periodicity
        self.input_normalization_mean = normalization_mean
        self.input_normalization_std = normalization_std
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.context = context if context is not None else {}  # additional arbitrary metadata

        # (None, x, y) => (1, x, y)
        if self.input_shape[0] is None:
            tmp = list(self.input_shape)
            tmp[0] = 1
            self.input_shape = tuple(tmp)
        if self.output_shape[0] is None:
            tmp = list(self.output_shape)
            tmp[0] = 1
            self.output_shape = tuple(tmp)

    @property
    def input_attributes(self):
        return self.input_shape[2]

    @property
    def input_length(self):
        return self.input_shape[1]

    @property
    def output_attributes(self):
        return self.output_shape[2]

    @property
    def output_length(self):
        return self.output_shape[1]

    @property
    def output_normalization_mean(self) -> List[float]:
        return [self.input_normalization_mean[i] for i in self.input_to_output_indices]

    @property
    def output_normalization_std(self) -> List[float]:
        return [self.input_normalization_std[i] for i in self.input_to_output_indices]

    def __repr__(self) -> str:
        return f'ModelMetadata(uuid={self.uuid},features={self.input_features},periodicity={self.periodicity},' \
               f'normalization_mean={self.input_normalization_mean},normalization_std={self.input_normalization_std},' \
               f'input_shape={self.input_shape},output_shape={self.output_shape})'

    def deepcopy(self) -> 'ModelMetadata':
        return ModelMetadata(self.uuid,
                             self.input_features.copy(),
                             self.input_shape,
                             self.output_shape,
                             self.periodicity.copy(),
                             self.input_normalization_mean.copy(),
                             self.input_normalization_std.copy(),
                             self.output_features.copy(),
                             self.context.copy())

    def to_dict(self) -> dict:
        """ModelMetadata.from_dict(json.loads(json.dumps(model_metadata.to_dict()))) is idempotent."""
        return {
            'uuid': self.uuid,
            'input_features': self.input_features,
            'output_features': self.output_features,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'periodicity': self.periodicity,
            'normalization_mean': list(map(lambda x: float(x), self.input_normalization_mean)),
            'normalization_std': list(map(lambda x: float(x), self.input_normalization_std)),
            'context': self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelMetadata':
        """ModelMetadata.from_dict(json.loads(json.dumps(model_metadata.to_dict()))) is idempotent."""
        return cls(uuid=d.get('uuid'),
                   input_features=d.get('input_features'),
                   output_features=d.get('output_features'),
                   input_shape=tuple(d.get('input_shape')),
                   output_shape=tuple(d.get('output_shape')),
                   periodicity=d.get('periodicity'),
                   normalization_mean=d.get('normalization_mean'),
                   normalization_std=d.get('normalization_std'),
                   context=d.get('context'))

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, separators=(',', ':'))  # no whitespace to minimize file size

    @classmethod
    def load(cls, path: str) -> 'ModelMetadata':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the given DataFrame according to the metadata (inplace)."""
        for i, feature in enumerate(self.input_features):
            df[feature] = (df[feature] - self.input_normalization_mean[i]) / self.input_normalization_std[i]
        return df
