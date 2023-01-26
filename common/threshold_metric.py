from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ThresholdMetric(ABC):
    """
    Depending on the use case, threshold metrics can take various forms.
    The absolute difference between actual and predicted value might suffice, or thresholds differ for every attribute.
    By implementing the abstract ThresholdMetric class, SenseReduce can consider custom threshold metrics.
    Threshold metric implementations must be stateless.
    When adding new threshold metrics, do not forget to extend the ThresholdMetricFactory.
    """

    _builders = {}

    @abstractmethod
    # TODO: for more complex thresholds, the datetime will be relevant
    def is_threshold_violation(self, measurement: np.array, prediction: np.array) -> bool:
        """

        Args:
            measurement: the measured attributes, indexed by their feature name
            prediction: the expected value, indexed by their feature name

        Returns:
            A boolean indicating whether the predicted value lies within the threshold.
        """
        pass

    @abstractmethod
    # TODO: for more complex thresholds, the datetime will be relevant
    def threshold_score(self, measurement: np.array, prediction: np.array) -> float:
        """An optional function that can be used to compare and rank different predictions.

        Args:
            measurement: the measured attributes, indexed by their feature name
            prediction: the expected value, indexed by their feature name

        Returns:
            A float indicating the quality of the prediction: the lower, the better.
        """
        return 0.0

    @abstractmethod
    def to_dict(self) -> dict:
        """Serializes a ThresholdMetric object to a dictionary.

        If dictionary d is returned, ThresholdMetricFactory.create_threshold_metric(d) should return an equal instance
        of self.

        Returns:
            A dictionary where the key 'type' indicates the class and 'object' contains metric-specific data.
        """
        return {
            'type': self.__class__.__name__,
            'object': self.__dict__
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ThresholdMetric':
        if d.get('type') is None or d.get('object') is None:
            raise ValueError('Malformed threshold dictionary: missing type or object field.')
        builder = cls._builders.get(d['type'])
        if builder is None:
            raise ValueError(f'Unsupported threshold type: {d["type"]}')
        return builder(**d['object'])

    @classmethod
    def register_type(cls, type_name: str, builder):
        """Registers a builder function for a given threshold metric.

        Args:
            type_name: The name of the threshold metric.
            builder: A function that takes a dictionary as input and returns a threshold object.
        """
        cls._builders[type_name] = builder


class ConjunctiveThreshold(ThresholdMetric):
    """A threshold metric that is violated if all given metrics are violated."""

    def __init__(self, threshold_metrics: List[ThresholdMetric]):
        self.threshold_metrics = threshold_metrics

    def __repr__(self) -> str:
        return f'{ConjunctiveThreshold.__name__}(threshold_metrics={self.threshold_metrics})'

    def is_threshold_violation(self, measurement: np.array, prediction: np.array) -> bool:
        return all(metric.is_threshold_violation(measurement, prediction) for metric in self.threshold_metrics)

    def threshold_score(self, measurement: np.array, prediction: np.array) -> float:
        return min(metric.threshold_score(measurement, prediction) for metric in self.threshold_metrics)

    def to_dict(self) -> dict:
        return {
            'type': ConjunctiveThreshold.__name__,
            'object': {
                'threshold_metrics': [metric.to_dict() for metric in self.threshold_metrics]
            }
        }


class DisjunctiveThreshold(ThresholdMetric):
    """Combines multiple metrics into a single metric that is violated if any of the sub-metrics is violated."""

    def __init__(self, threshold_metrics: List[ThresholdMetric]):
        self.threshold_metrics = threshold_metrics

    def __repr__(self) -> str:
        return f'{DisjunctiveThreshold.__name__}(threshold_metrics={self.threshold_metrics})'

    def is_threshold_violation(self, measurement: np.array, prediction: np.array) -> bool:
        return any(metric.is_threshold_violation(measurement, prediction) for metric in self.threshold_metrics)

    def threshold_score(self, measurement: np.array, prediction: np.array) -> float:
        return max(metric.threshold_score(measurement, prediction) for metric in self.threshold_metrics)

    def to_dict(self) -> dict:
        return {
            'type': DisjunctiveThreshold.__name__,
            'object': {
                'threshold_metrics': [metric.to_dict() for metric in self.threshold_metrics]
            }
        }


class L2Threshold(ThresholdMetric):
    """Computes the target metric as the Euclidian distance (L2 norm) between measurement and prediction."""

    def __init__(self,
                 threshold: float,
                 measurement_indices: List[int],
                 prediction_indices: List[int]
                 ) -> None:
        self.threshold = threshold
        self.m_indices = measurement_indices
        self.p_indices = prediction_indices

    def __repr__(self) -> str:
        return f'{L2Threshold.__name__}(threshold={self.threshold}, ' \
               f'measurement_indices={self.m_indices}, prediction_indices={self.p_indices})'

    def is_threshold_violation(self, measurement: np.array, prediction: np.array) -> bool:
        return self.threshold_score(measurement, prediction) > self.threshold

    def threshold_score(self, measurement: np.array, prediction: np.array) -> float:
        return np.linalg.norm(measurement[self.m_indices] - prediction[self.p_indices])

    def to_dict(self) -> dict:
        return {
            'type': L2Threshold.__name__,
            'object': {
                'threshold': self.threshold,
                'measurement_indices': self.m_indices,
                'prediction_indices': self.p_indices
            }
        }


ThresholdMetric.register_type(ConjunctiveThreshold.__name__, ConjunctiveThreshold)
ThresholdMetric.register_type(DisjunctiveThreshold.__name__, DisjunctiveThreshold)
ThresholdMetric.register_type(L2Threshold.__name__, L2Threshold)
