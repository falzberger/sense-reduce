import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union

import pandas as pd
import tensorflow as tf

from base.model import Model
from base.node import NodeID
from base.training import mse_weighted
from base.window_generator import WindowGenerator
from common.utils import convert_datetime, normalize_df, split_df


class LearningStrategy(ABC):
    """An interface for defining strategies for continual learning of machine learning models.

    By subclassing this class, new strategies can be implemented. Strategies for retraining and transfer learning are
    already implemented. In the future, strategies for other continual learning methods will be added.

    For example:
      - Random Replay: Updates a model with the updates and a random selection of the initial data
      - Buffered Replay: Maintains a limited buffer containing initial training data and updates with reservoir sampling
    """

    _builders = {}

    @abstractmethod
    def add_node(self, node_id: NodeID, initial_df: pd.DataFrame) -> None:
        """Adds a new node to the strategy and its associated initial data (can be empty)."""
        pass

    @abstractmethod
    def add_model(self, model: Model) -> None:
        """Adds a new model to the strategy that can be considered for future training."""
        pass

    @abstractmethod
    def add_new_measurements(self, node_id: NodeID, new_measurements: pd.DataFrame) -> None:
        """Called if there are new measurements for the node with the given id. Also called to register a new node.

        The LearningStrategy shall decide how to maintain the data of a node.
        Depending on the implementation, this may involve removing old data or adding new data.
        """
        pass

    @abstractmethod
    def get_candidate_models(self, node_id: NodeID) -> List[Model]:
        """Returns a list of models maintained by the strategy for the specified node.

        The concrete implementation can decide whether models are maintained for each node individually or globally.
        """
        pass

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the strategy.

        If dictionary d is returned, LearningStrategy.from_dict(d) should return an equal instance of self.

        Returns:
            A dictionary where the key 'type' indicates the class and 'object' contains strategy-specific data.
        """
        return {
            'type': self.__class__.__name__,
            'object': self.__dict__
            }

    @classmethod
    def from_dict(cls, d: dict) -> 'LearningStrategy':
        if d.get('type') is None or d.get('object') is None:
            raise ValueError('Malformed strategy dictionary: missing type or object field.')
        builder = cls._builders.get(d['type'])
        if builder is None:
            raise ValueError(f'Unsupported strategy type: {d["type"]}')
        return builder(**d['object'])

    def copy(self) -> 'LearningStrategy':
        """Creates a (shallow) copy of the strategy without considering the internal state."""
        return LearningStrategy.from_dict(self.to_dict())

    @classmethod
    def register_type(cls, type_name: str, builder):
        """Registers a builder function for a given strategy type.

        Args:
            type_name: The name of the strategy type.
            builder: A function that takes a dictionary as input and returns an instance of the strategy.
        """
        cls._builders[type_name] = builder


class NoUpdateStrategy(LearningStrategy):
    """A wrapper for a non-continual model that does not execute any update."""

    def __init__(self) -> None:
        super().__init__()
        self.models = []

    def __repr__(self) -> str:
        return 'NoUpdateStrategy()'

    def to_dict(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'object': {}
            }

    def add_node(self, node_id: NodeID, initial_df: pd.DataFrame) -> None:
        return None

    def add_model(self, model: Model) -> None:
        self.models.append(model)

    def get_candidate_models(self, node_id: NodeID) -> List[Model]:
        return self.models

    def add_new_measurements(self, node_id: NodeID, new_measurements: pd.DataFrame) -> None:
        return None


class TransferLearningStrategy(LearningStrategy):
    """Applies transfer learning to the model by freezing the defined layers for training.

    New measurements are kept in a temporary buffer that is emptied after training a new model.
    The strategy starts with an empty buffer and a model that is trained on the initial data.
    If the freeze_layers list is empty, all weights remain trainable. Hence, a fine-tuning strategy is applied.

    Args:
        epochs: Maximum number of epochs to train the model.
        optimizer: Optimizer to use for training.
        learning_rate: Learning rate for the optimizer.
        stride: Stride of the sliding window used for training.
        validation: Duration of last data to use for validation, e.g., '4w' for 4 weeks.
    """

    def __init__(self,
                 freeze_layers: List[Union[str, int]],
                 epochs: int = 100,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 stride: int = 1,
                 validation: Optional[str] = None,
                 ) -> None:
        self.freeze_layers = freeze_layers
        self.epochs = epochs
        self.optimizer_str = optimizer
        self.learning_rate = float(learning_rate)
        self.stride = int(stride)

        self.validation = None if validation is None else pd.Timedelta(validation)

        self.base_model: Optional[Model] = None
        self.node_id_to_model: Dict[NodeID, Model] = {}
        self.node_id_to_data: Dict[NodeID, pd.DataFrame] = {}
        self.node_id_to_changed: Dict[NodeID, bool] = {}

    def __repr__(self) -> str:
        return f'TransferLearningStrategy({self.to_dict()})'

    def _get_optimizer(self) -> tf.optimizers.Optimizer:
        if self.optimizer_str == 'adam':
            return tf.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_str == 'rmsprop':
            return tf.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer_str}')

    def to_dict(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'object': {
                'freeze_layers': self.freeze_layers,
                'epochs': self.epochs,
                'optimizer': self.optimizer_str,
                'learning_rate': self.learning_rate,
                'stride': self.stride,
                'validation': None if self.validation is None else str(self.validation),
                }
            }

    def _retrain_node_model(self, node_id: NodeID):
        model = self.node_id_to_model[node_id]
        df = self.node_id_to_data[node_id]
        old_weights = model.model.get_weights()

        md = model.metadata
        convert_datetime(df, md.periodicity)

        if self.validation is not None:
            train_df = df.loc[:df.index[-1] - self.validation, :].copy()
            val_df = df.loc[df.index[-1] - self.validation:, :].copy()
            md.apply_normalization(val_df)
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=int(self.epochs / 6),
                                                     mode='min',
                                                     factor=0.2,
                                                     verbose=1
                                                     ),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=int(self.epochs / 4),
                                                 restore_best_weights=True
                                                 )]
        else:
            train_df = df.copy()
            val_df = None
            callbacks = None
        # we must not change the normalization parameters of the base model
        md.apply_normalization(train_df)

        window = WindowGenerator(md.input_length, md.output_length,
                                 stride=self.stride,
                                 sampling_rate=1,  # TODO: support sub-hourly data resolution
                                 train_df=train_df, val_df=val_df, test_df=None,
                                 norm_mean=md.input_normalization_mean, norm_std=md.input_normalization_std,
                                 periodicity=md.periodicity,
                                 input_features=md.input_features,
                                 output_features=md.output_features
                                 )
        model.model.fit(window.train, validation_data=window.val, epochs=self.epochs, callbacks=callbacks, verbose=2)

        if self.validation is not None:
            new_weights = model.model.get_weights()
            new = model.model.evaluate(window.val)

            model.model.set_weights(old_weights)
            old = model.model.evaluate(window.val)

            if new[0] < old[0]:  # compare loss
                # found a better model, remove all except validation data
                model.model.set_weights(new_weights)
                self.node_id_to_data[node_id] = val_df.copy()
            else:
                logging.info('No improvement found, keeping old model.')
        else:
            # if we don't have validation data, we can't compare the models and remove all data
            self.node_id_to_data[node_id] = pd.DataFrame(columns=df.columns)
        return model

    def add_model(self, model: Model) -> None:
        if self.base_model is None:
            for layer in self.freeze_layers:
                logging.debug(f'Freezing layer in new model: {layer}')
                model.model.get_layer(layer).trainable = False
            self.base_model = model
        else:
            logging.warning('TransferLearningStrategy only supports one base model. Ignoring additional model.')

    def add_node(self, node_id: NodeID, initial_df: pd.DataFrame) -> None:
        if self.base_model is None:
            raise Exception('No base model set.')

        # we only use the continual data for transfer learning
        self.node_id_to_data[node_id] = pd.DataFrame(columns=initial_df.columns)
        self.node_id_to_model[node_id] = self.base_model.clone()
        self.node_id_to_model[node_id].model.compile(
            optimizer=self._get_optimizer(),
            loss=mse_weighted,
            metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
            )

        # for new nodes, we do not retrain the model immediately
        self.node_id_to_changed[node_id] = False

    def add_new_measurements(self, node_id: NodeID, new_measurements: pd.DataFrame) -> None:
        self.node_id_to_data[node_id] = pd.concat([self.node_id_to_data.get(node_id), new_measurements])
        self.node_id_to_changed[node_id] = True

    def get_candidate_models(self, node_id: NodeID) -> List[Model]:
        if node_id not in self.node_id_to_model:
            raise ValueError(f'No model found for node {node_id}.')
        else:
            if self.node_id_to_changed[node_id]:
                self._retrain_node_model(node_id)
                self.node_id_to_changed[node_id] = False
            return [self.node_id_to_model[node_id]]


class RetrainStrategy(LearningStrategy):
    """Retrains a model on all available data (initial + updates), using early stopping and train-validation split.

    If no validation argument is given, the model is trained the specified number of epochs without validation.
    On each training, a new model is created (with newly initialized weights) and trained on all collected data.
    The new model is always returned (not comparing its performance to the previous model).

    Args:
        epochs: Maximum number of epochs to train the model.
        patience: Number of epochs to wait for improvement of loss function before early stopping.
        optimizer: Optimizer to use for training.
        learning_rate: Learning rate for the optimizer.
        stride: Stride of the sliding window used for training.
        validation: Duration of last data to use for validation, e.g., '4w' for 4 weeks or 0.2 for 20% of the data.
    """

    def __init__(self,
                 epochs: int = 100,
                 patience: int = 20,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 stride: int = 1,
                 validation: Optional[Union[str, float]] = None,
                 use_initial_data: bool = True,
                 ) -> None:
        self.max_epochs = epochs
        self.patience = patience
        self.optimizer_str = optimizer
        self.learning_rate = float(learning_rate)
        self.stride = int(stride)
        self.use_initial_data = use_initial_data

        if validation is None:
            self.validation = None
        else:
            try:
                self.validation = float(validation)
                assert 0 < self.validation < 1, 'Validation split must be a float between 0 and 1.'
            except ValueError:
                self.validation = pd.Timedelta(validation)

        self.base_model: Optional[Model] = None
        self.node_id_to_model: Dict[NodeID, Model] = {}
        self.node_id_to_data: Dict[NodeID, pd.DataFrame] = {}
        self.node_id_to_changed: Dict[NodeID, bool] = {}

    def __repr__(self) -> str:
        return f'RetrainStrategy({self.to_dict()})'

    def to_dict(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'object': {
                'epochs': self.max_epochs,
                'patience': self.patience,
                'optimizer': self.optimizer_str,
                'learning_rate': self.learning_rate,
                'stride': self.stride,
                'validation': None if self.validation is None else str(self.validation),
                'use_initial_data': self.use_initial_data,
                }
            }

    def _get_optimizer(self) -> tf.optimizers.Optimizer:
        if self.optimizer_str == 'adam':
            return tf.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_str == 'rmsprop':
            return tf.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer_str}')

    def _retrain_node_model(self, node_id: NodeID):
        old_model = self.node_id_to_model[node_id].model
        md = self.node_id_to_model[node_id].metadata
        df = self.node_id_to_data[node_id]
        convert_datetime(df, md.periodicity)

        if self.validation is None:
            train_df = df.copy()
            val_df = None
            cb = None
        else:
            cb = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=int(self.patience / 2),
                                                     mode='min',
                                                     factor=0.2,
                                                     verbose=1
                                                     ),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)]
            if isinstance(self.validation, float):
                train_df, val_df, _ = split_df(df, 1 - self.validation, self.validation)
            else:  # self.validation is a timedelta
                train_df = df.loc[:df.index[-1] - self.validation, :].copy()
                val_df = df.loc[df.index[-1] - self.validation:, :].copy()

        # adjust the normalization values to the new data
        norm_mean, norm_std = normalize_df(md.input_features, train_df, [] if val_df is None else [val_df])
        window = WindowGenerator(md.input_length, md.output_length, self.stride,
                                 sampling_rate=1,  # TODO: support non-hourly data resolution
                                 train_df=train_df, val_df=val_df, test_df=None,
                                 norm_mean=norm_mean, norm_std=norm_std,
                                 periodicity=md.periodicity,
                                 input_features=md.input_features,
                                 output_features=md.output_features
                                 )

        # reset the model weights
        tf.keras.backend.clear_session()
        new_model = tf.keras.models.clone_model(old_model)
        new_model.compile(
            optimizer=self._get_optimizer(),
            loss=mse_weighted,
            metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
            )
        new_model.fit(window.train, validation_data=window.val, epochs=self.max_epochs, callbacks=cb, verbose=2)

        # if the model did not improve, do not change
        md = md.deepcopy()
        # if self.validation is not None:
        #     new = new_model.evaluate(window.val)
        #     old = old_model.evaluate(window.val)
        #
        #     if new[0] < old[0]:  # only adapt if loss function improved
        #         md.input_normalization_mean = norm_mean
        #         md.input_normalization_std = norm_std
        #         self.node_id_to_model[node_id] = Model(new_model, md)
        #     else:
        #         logging.info('No improvement found, keeping old model.')
        md.input_normalization_mean = norm_mean
        md.input_normalization_std = norm_std
        self.node_id_to_model[node_id] = Model(new_model, md)

    def add_model(self, model: Model) -> None:
        if self.base_model is None:
            self.base_model = model
        else:
            logging.warning('RetrainStrategy only supports one base model. Ignoring additional model.')

    def add_node(self, node_id: NodeID, initial_df: pd.DataFrame) -> None:
        if self.base_model is None:
            raise Exception('No base model set.')

        if self.use_initial_data:
            self.node_id_to_data[node_id] = initial_df
        else:
            self.node_id_to_data[node_id] = pd.DataFrame(columns=initial_df.columns)
        self.node_id_to_model[node_id] = self.base_model
        # for new nodes, we do not retrain the model immediately
        self.node_id_to_changed[node_id] = False

    def add_new_measurements(self, node_id: NodeID, new_measurements: pd.DataFrame) -> None:
        self.node_id_to_data[node_id] = pd.concat([self.node_id_to_data.get(node_id), new_measurements])
        self.node_id_to_changed[node_id] = True

    def get_candidate_models(self, node_id: NodeID) -> List[Model]:
        if node_id not in self.node_id_to_model:
            raise ValueError(f'No model found for node {node_id}.')
        else:
            if self.node_id_to_changed[node_id]:
                self._retrain_node_model(node_id)
                self.node_id_to_changed[node_id] = False
            return [self.node_id_to_model[node_id]]


LearningStrategy.register_type(NoUpdateStrategy.__name__, NoUpdateStrategy)
LearningStrategy.register_type(TransferLearningStrategy.__name__, TransferLearningStrategy)
LearningStrategy.register_type(RetrainStrategy.__name__, RetrainStrategy)
