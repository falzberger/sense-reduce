from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from base.model import Model
from base.node import NodeID


# TODO: currently unused, should be used as a separate process to manage models on the base station
class ModelManager(ABC):
    """
    An abstract class describing the interfaces for valid model managers.

    The model manager is responsible for the management of the models on the base station.
    It decides when and how to update maintained models, and which models to provide to
    the node manager as deployment candidates.
    """

    _builders = {}

    @abstractmethod
    def get_candidate_models(self, node_id: NodeID) -> List[Model]:
        """
        Returns a list of models that are available for deployment to the node with the given id.
        """
        pass

    @abstractmethod
    def on_new_measurements(self, node_id: NodeID, df: pd.DataFrame) -> None:
        """
        Called when new measurements are available for a node.

        Args:
            node_id: The id of the node.
            df: The new measurements.
        """
        pass

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the model manager, without its current state.

        Returns:
            A dictionary where the key 'type' indicates the class and 'object' contains __init__ **kwargs.
        """
        return {
            'type': self.__class__.__name__,
            'object': self.__dict__
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelManager':
        if d.get('type') is None or d.get('object') is None:
            raise ValueError('Malformed dictionary: missing type or object field.')
        builder = cls._builders.get(d['type'])
        if builder is None:
            raise ValueError(f'Unsupported model manager type: {d["type"]}')
        return builder(**d['object'])

    @classmethod
    def register_type(cls, type_name: str, builder):
        """Registers a builder function for a given model manager type.

        Args:
            type_name: The name of the model manager type.
            builder: A function that takes the same arguments as the model manager's __init__ method
            and returns an instance of the model manager.
        """
        cls._builders[type_name] = builder


class NoUpdateManager(ModelManager):
    """A model manager that never updates the models."""

    def get_candidate_models(self, node_id: NodeID) -> List[Model]:
        return []

    def on_new_measurements(self, node_id: NodeID, df: pd.DataFrame) -> None:
        pass


ModelManager.register_type(NoUpdateManager.__name__, NoUpdateManager)
