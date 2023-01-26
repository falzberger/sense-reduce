import functools
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from base.learning_strategy import LearningStrategy
from base.deployment_strategy import DeploymentStrategy
from base.model import Model
from base.node import Node, NodeID
from common import Predictor, DataStorage, ModelMetadata, LiteModel, ThresholdMetric


class NodeManager:
    """Manages multiple sensor nodes using a continual learning strategy and a deployment strategy.

    For each node, the manager maintains the exact same `Predictor` as currently deployed on node.
    By using it, the manager can return the same predictions as the node currently has.

    The models are updated according to the manager's `LearningStrategy`.
    Updates are triggered when new data points for the respective node are added to the manager.
    Whether a registered node receives a model update or not is decided by the manager's DeploymentStrategy.

    Future work:
     - In the long run, an asynchronous "model manager" should be implemented that takes care of model updates,
       otherwise the manager will be blocked by the model training of the continual learning strategy.
    """

    FILE_NAME = 'node_manager.json'

    def __init__(self,
                 cl_strategy: LearningStrategy,
                 deploy_strategy: DeploymentStrategy,
                 working_dir: str = '.'
                 ) -> None:
        self.cl_strategy = cl_strategy
        self.deploy_strategy = deploy_strategy
        self.working_dir = working_dir  # temporary working storage

        self._node_ids_to_node: Dict[str, Node] = dict()
        self._node_has_lite_model: Dict[str, bool] = dict()

    @property
    def node_ids(self) -> List[NodeID]:
        return list(self._node_ids_to_node.keys())

    def get_node_dir(self, node_id: NodeID) -> str:
        """Returns the directory where the models of the specified node are stored."""
        return os.path.join(self.working_dir, node_id)

    def get_predictor(self, node_id: NodeID) -> Predictor:
        return self.get_node(node_id).predictor

    def get_cl_model(self, node_id: NodeID) -> Model:
        return self.get_node(node_id).cl_model

    def get_horizon_updates(self, node_id: NodeID) -> pd.DatetimeIndex:
        return self.get_node(node_id).horizon_updates

    def get_threshold_violations(self, node_id: NodeID) -> DataStorage:
        return self.get_node(node_id).threshold_violations

    def get_model_deployments(self, node_id: NodeID) -> pd.DataFrame:
        return self.get_node(node_id).model_deployments

    def get_node(self, node_id: NodeID) -> Node:
        return self._node_ids_to_node[node_id]

    def get_last_synchronization(self, node_id) -> datetime:
        return self.get_node(node_id).last_synchronization

    def get_prediction_at(self, node_id: NodeID, dt: datetime) -> Optional[pd.Series]:
        """Returns the predicted temperature at the sensor node for the specified datetime or None if the datetime
        is not in range of the current prediction horizon. """
        return self.get_node(node_id).get_prediction_at(dt)

    def add_node(self,
                 node_id: NodeID,
                 threshold_metric: ThresholdMetric,
                 initial_df: pd.DataFrame,
                 start_dt: datetime,
                 lite_model=True,
                 ) -> Node:
        """Creates a new node with the given id and threshold metric and adds it to the manager.

        Args:
            node_id: The id of the node.
            threshold_metric: The threshold metric for the node.
            initial_df: The initial data for the first predictions of the node.
            start_dt: When to start the first prediction horizon. If not within one hour after initial_df,
                yearly-averaged data from initial_df is used.
            lite_model: Whether to use a TFlite model or a full model.
        """
        if node_id in self._node_ids_to_node:
            logging.warning(f'Node with ID {node_id} has already been added to NodeManager')
        self.cl_strategy.add_node(node_id, initial_df.copy())
        model = self.deploy_strategy.on_initial_deployment(
            functools.partial(self.cl_strategy.get_candidate_models, node_id),
            initial_df.copy(), start_dt)

        if start_dt > initial_df.index[-1] + timedelta(hours=1):
            logging.info(f'Generating initial data for node {node_id} to match start datetime {start_dt}')
            initial_data = DataStorage.from_previous_years_average(
                start_dt - timedelta(hours=model.metadata.input_length),
                start_dt,
                initial_df,
                model.metadata.output_features,
            )
        else:
            initial_data = DataStorage.from_data(initial_df.copy(),
                                                 pd.DataFrame(columns=model.metadata.output_features, dtype=np.float64))

        model.save_and_convert(self.get_node_dir(node_id))
        if lite_model:
            predictor_model = LiteModel.load(self.get_node_dir(node_id))
        else:
            predictor_model = model
        predictor = Predictor(predictor_model, initial_data.copy())
        predictor.update_prediction_horizon(start_dt)
        node = Node(node_id, threshold_metric, model, predictor)
        self._node_ids_to_node[node_id] = node
        self._node_has_lite_model[node_id] = lite_model
        return node

    def on_threshold_violation(self,
                               node_id: NodeID,
                               dt: datetime,
                               measurement: np.ndarray,
                               new_measurements: pd.DataFrame
                               ) -> Optional[ModelMetadata]:
        """Called when a node reports a threshold violation. Persists associated data and updates the node's
        predictions.

        Args:
            node_id: The id of the node.
            dt: The datetime of the threshold violation.
            measurement: The measurement that caused the threshold violation.
            new_measurements: The new measurements that were received by the node.
                TODO: do we always need hourly resolution here?

        Returns:
            The metadata of the new model or None if no model update required.
        """
        n = self.get_node(node_id)
        n.add_measurement_df(new_measurements)

        prediction = n.get_prediction_at(dt)
        n.add_threshold_violation(dt, measurement, prediction.to_numpy())
        logging.debug(f'Threshold violation ({dt}): predicted={prediction.to_numpy()}, measured={measurement}')

        self.cl_strategy.add_new_measurements(node_id, new_measurements.copy())
        deploy_model = self.deploy_strategy.on_threshold_violation(
            n, dt, functools.partial(self.cl_strategy.get_candidate_models, node_id))
        if deploy_model is None:
            n.add_prediction_df(n.predictor.get_predictions_until(dt))
            n.predictor.update_prediction_horizon(dt)
            n.last_synchronization = dt
            # use previous prediction to correct the new horizon
            n.predictor.adjust_to_measurement(dt, measurement, n.get_prediction_at(dt).to_numpy())
            return None
        else:
            logging.debug(f'New model for deployment after threshold violation for node {node_id}')
            deploy_model.save_and_convert(self.get_node_dir(node_id))
            # the node manager will update its model after it has been fetched
            return deploy_model.metadata

    def on_horizon_update(self, node_id: NodeID, dt: datetime, new_measurements: pd.DataFrame
                          ) -> Optional[ModelMetadata]:
        """Called for horizon updates after an uninterrupted prediction horizon has passed.

        Args:
            node_id: The ID of the node.
            dt: The datetime of the horizon update, i.e., when the new prediction horizon starts.
            new_measurements: The hourly measurements that happened in this horizon.

        Returns:
            The metadata of the new model or None if no model update required.
        """
        n = self.get_node(node_id)
        n.add_measurement_df(new_measurements)
        n.add_horizon_update(dt)
        logging.debug(f'Horizon update for node {node_id}')

        self.cl_strategy.add_new_measurements(node_id, new_measurements.copy())
        deploy_model = self.deploy_strategy.on_horizon_update(
            n, dt, functools.partial(self.cl_strategy.get_candidate_models, node_id))
        if deploy_model is None:
            # we do this in on_model_deployment() otherwise
            n.add_prediction_df(n.predictor.get_predictions_until(dt))
            n.predictor.update_prediction_horizon(dt)
            n.last_synchronization = dt
            return None
        else:
            logging.debug(f'New model for deployment after horizon update for node {node_id}')
            deploy_model.save_and_convert(self.get_node_dir(node_id))
            # the node manager will update its model after it has been fetched
            return deploy_model.metadata

    def on_model_deployment(self, node_id: NodeID, dt: datetime) -> str:
        """Called when a node requested a new model. Updates the node's predictor in the node manager.

        Inherently, we run into a distributed synchronization problem here. We update the node's
        predictor (and prediction horizon), but the node itself is still using the old model until
        it fetched and deployed the new model from the base station. However, we assume that this
        error is negligible: in the worst case, we get an additional threshold violation.

        Args:
            node_id: The id of the node.
            dt: The datetime of the deployment request.

        Returns:
            The path to the new LiteModel for the node.
        """
        logging.debug(f'Deploying new model for node {node_id}')

        node_dir = self.get_node_dir(node_id)
        cl_model = Model.load(node_dir)

        n = self.get_node(node_id)
        n.add_prediction_df(n.predictor.get_predictions_until(dt))
        if self._node_has_lite_model[node_id]:
            n.predictor.set_model(LiteModel.load(node_dir), dt)
        else:
            n.predictor.set_model(cl_model, dt)
        n.cl_model = cl_model
        n.last_synchronization = dt

        # FIXME: this is not correct if we do not use a lite model
        l_model_path = os.path.join(node_dir, LiteModel.FILE_NAME)
        n.add_model_deployment(dt, os.path.getsize(l_model_path))
        return l_model_path

    def save(self, path='.') -> None:
        """Saves a representation of the node manager's data to the specified directory."""
        try:
            os.makedirs(path)
        except FileExistsError:
            logging.warning(f'Directory "{path}" already exists. Overwriting data.')

        with open(os.path.join(path, NodeManager.FILE_NAME), 'w') as f:
            json.dump({
                'cl_strategy': self.cl_strategy.to_dict(),
                'deploy_strategy': self.deploy_strategy.to_dict(),
                'node_ids': self.node_ids,
            }, f, indent=4)

        for node_id in self.node_ids:
            node_dir = os.path.join(path, node_id)
            os.makedirs(node_dir, exist_ok=True)
            self.get_node(node_id).save(node_dir)

    @classmethod
    def load(cls, path='.', working_dir='.', lazy_loading=False) -> 'NodeManager':
        """Loads a node manager from the specified directory."""
        with open(os.path.join(path, NodeManager.FILE_NAME), 'r') as f:
            config = json.load(f)

        nm = cls(
            LearningStrategy.from_dict(config['cl_strategy']),
            DeploymentStrategy.from_dict(config['deploy_strategy']),
            working_dir
        )

        for node_id in config['node_ids']:
            node_dir = os.path.join(path, node_id)
            node = Node.load(path=node_dir, lazy_loading=lazy_loading)
            nm._node_ids_to_node[node_id] = node

        return nm
