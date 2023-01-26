from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Callable

import pandas as pd

from base.model import Model
from base.node import Node
from common import PredictionHorizon


class DeploymentStrategy(ABC):
    """
    An abstract class describing the interfaces for valid model deployment strategies.

    In general, SenseReduce uses a 2-step pull-based update mechanism: First, a sensor node has to ask the base station
    whether an updated model is available. Then the base station decides whether the deployment of an updated model
    pays off according to the utilized deployment strategy. If the sensor node should update its model, it receives
    the metadata and the location of the new model, otherwise an application-specific status code.

    More sophisticated ideas for deployment strategies that we may implement in the future:
    - Linear Programming: incorporate prospective energy usage of data transfer (using size of new model) with the
      possible gains through reduced threshold violations?
    - Machine Learning decision through reinforcement learning
    """

    _builders = {}

    @abstractmethod
    def on_initial_deployment(self,
                              candidate_models: Callable[[], List[Model]],
                              initial_df: pd.DataFrame,
                              dt: datetime
                              ) -> Model:
        """Called to determine the initial model for a node. Must return a Model instance. """
        pass

    @abstractmethod
    def on_threshold_violation(self,
                               node: Node,
                               dt: datetime,
                               candidate_models: Callable[[], List[Model]],
                               ) -> Optional[Model]:
        """Called when a threshold violation occurs.

        Returns:
             The new model to deploy or None if no update is needed.
        """
        pass

    @abstractmethod
    def on_horizon_update(self,
                          node: Node,
                          dt: datetime,
                          candidate_models: Callable[[], List[Model]],
                          ) -> Optional[Model]:
        """Called when a node has reached the end of its current prediction horizon.

        Returns:
            The new model to deploy or None if no update is needed.
        """
        pass

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the strategy.

        If dictionary d is returned, DeploymentStrategy.from_dict(d) should return an equal instance of self.

        Returns:
            A dictionary where the key 'type' indicates the class and 'object' contains strategy-specific data.
        """
        return {
            'type': self.__class__.__name__,
            'object': self.__dict__
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DeploymentStrategy':
        if d.get('type') is None or d.get('object') is None:
            raise ValueError('Malformed strategy dictionary: missing type or object field.')
        builder = cls._builders.get(d['type'])
        if builder is None:
            raise ValueError(f'Unsupported strategy type: {d["type"]}')
        return builder(**d['object'])

    def copy(self) -> 'DeploymentStrategy':
        """Creates a (shallow) copy of the strategy without considering the internal state."""
        return DeploymentStrategy.from_dict(self.to_dict())

    @classmethod
    def register_type(cls, type_name: str, builder):
        """Registers a builder function for a given strategy type.

        Args:
            type_name: The name of the strategy type.
            builder: A function that takes a dictionary as input and returns an instance of the strategy.
        """
        cls._builders[type_name] = builder


class DeployOnceStrategy(DeploymentStrategy):

    def __repr__(self):
        return f'DeployOnceStrategy()'

    def on_initial_deployment(self, candidate_models: Callable[[], List[Model]], initial_df: pd.DataFrame, dt: datetime
                              ) -> Model:
        # TODO: devise a better algorithm for selecting best initial model
        return candidate_models()[0]

    def on_threshold_violation(self, node: Node, dt: datetime, candidate_models: Callable[[], List[Model]]
                               ) -> Optional[Model]:
        return None

    def on_horizon_update(self, node: Node, dt: datetime, candidate_models: Callable[[], List[Model]]
                          ) -> Optional[Model]:
        return None


class FixedIntervalStrategy(DeploymentStrategy):
    """A strategy that deploys a new model every fixed interval.

    The interval is only checked on threshold violations or horizon updates.
    I.e., in the worst case, the interval is exceeded by exactly one prediction horizon.

    Attributes:
        interval: The interval in seconds.
    """

    def __init__(self, interval: int):
        self.interval = interval
        self.last_deployment: Optional[datetime] = None

    def __repr__(self):
        return f'FixedIntervalStrategy(interval={self.interval})'

    def to_dict(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'object': {
                'interval': self.interval
            }
        }

    def on_initial_deployment(self, candidate_models: Callable[[], List[Model]], initial_df: pd.DataFrame, dt: datetime
                              ) -> Model:
        self.last_deployment = dt
        return candidate_models()[0]

    def on_threshold_violation(self, node: Node, dt: datetime, candidate_models: Callable[[], List[Model]]
                               ) -> Optional[Model]:
        if (dt - self.last_deployment).total_seconds() >= self.interval:
            self.last_deployment = dt
            return candidate_models()[0]
        return None

    def on_horizon_update(self, node: Node, dt: datetime, candidate_models: Callable[[], List[Model]]
                          ) -> Optional[Model]:
        if (dt - self.last_deployment).total_seconds() >= self.interval:
            self.last_deployment = dt
            return candidate_models()[0]
        return None


class CorrectiveStrategy(DeploymentStrategy):
    """Deploys an updated model if it had not led to a threshold violation at the time the current model has."""

    # TODO: more generic version? i.e., deploy if it would not have led to last 2 violations?

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self):
        return f'CorrectiveStrategy()'

    def on_initial_deployment(self, candidate_models: Callable[[], List[Model]], initial_df: pd.DataFrame, dt: datetime
                              ) -> Model:
        return candidate_models()[0]

    def on_threshold_violation(self, node: Node, dt: datetime, candidate_models: Callable[[], List[Model]]
                               ) -> Optional[Model]:
        measurement = node.threshold_violations.get_measurements().loc[dt]

        for model in candidate_models():
            input_df = node.data.get_measurements_previous_hours(node.last_synchronization, model.metadata.input_length)
            last_hour = input_df.index.max()
            prediction = model.predict(input_df)
            prediction.loc[last_hour] = input_df.loc[last_hour]
            ip = PredictionHorizon(prediction)

            model_pred = ip.get_prediction_at(dt)
            if not node.threshold_metric.is_threshold_violation(measurement, model_pred):
                # TODO: maybe rank by threshold_score?
                return model
        return None

    def on_horizon_update(self, node: Node, dt: datetime, candidate_models: Callable[[], List[Model]]
                          ) -> Optional[Model]:
        return None


DeploymentStrategy.register_type(DeployOnceStrategy.__name__, DeployOnceStrategy)
DeploymentStrategy.register_type(FixedIntervalStrategy.__name__, FixedIntervalStrategy)
# DeploymentStrategy.register_type(CorrectiveStrategy.__name__, CorrectiveStrategy)
