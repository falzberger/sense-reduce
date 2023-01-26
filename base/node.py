import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from base.model import Model
from common import ThresholdMetric, DataStorage, Predictor, LiteModel

NodeID = str


class Node:
    """A Node represents a single device that is connected to the base station."""

    def __init__(self,
                 uuid: NodeID,
                 threshold_metric: ThresholdMetric,
                 model: Model,
                 predictor: Predictor
                 ) -> None:
        self.uuid = uuid
        self.threshold_metric = threshold_metric
        self.cl_model = model
        self.predictor = predictor

        self._last_synchronization = self.data.get_measurements().index.max().to_pydatetime()

        self.threshold_violations = DataStorage(model.metadata.input_features,
                                                model.metadata.output_features)
        self.horizon_updates = pd.DatetimeIndex([])
        self.model_deployments = pd.DataFrame(
            columns=['size'],
            dtype=np.int64,
        )

    @property
    def data(self):
        """Collects all measurements and predictions for this node."""
        return self.predictor.data

    @property
    def last_synchronization(self) -> datetime:
        return self._last_synchronization

    @last_synchronization.setter
    def last_synchronization(self, dt: datetime):
        self._last_synchronization = dt

    def add_horizon_update(self, dt: datetime):
        self.horizon_updates = self.horizon_updates.insert(len(self.horizon_updates), dt)

    def add_threshold_violation(self, dt: datetime, measurement: np.ndarray, prediction: np.ndarray):
        self.threshold_violations.add_prediction(dt, prediction)
        self.threshold_violations.add_measurement(dt, measurement)

    def add_measurement(self, dt: datetime, values: np.ndarray):
        self.predictor.add_measurement(dt, values)

    def add_measurement_df(self, df: pd.DataFrame):
        self.predictor.add_measurement_df(df)

    def add_prediction(self, dt: datetime, values: np.ndarray):
        self.predictor.add_prediction(dt, values)

    def add_prediction_df(self, df: pd.DataFrame):
        self.predictor.add_prediction_df(df)

    def add_model_deployment(self, dt: datetime, size: int):
        self.model_deployments.loc[dt] = [size]

    def get_prediction_at(self, dt: datetime) -> pd.Series:
        return self.predictor.get_prediction_at(dt)

    def save(self, path='.') -> None:
        """Stores the node in a directory, using multiple files.

        The 'Predictor' class itself will not be saved. Its data is ephemeral and can be re-created when needed."""
        try:
            os.makedirs(path, exist_ok=False)
        except FileExistsError:
            logging.warning(f'Directory "{path}" already exists. Overwriting existing files.')

        self.cl_model.save_and_convert(path)
        self.data.save(path)

        with open(os.path.join(path, 'node.json'), 'w') as f:
            json.dump({
                'uuid': self.uuid,
                'threshold_metric': self.threshold_metric.to_dict(),
                'last_synchronization': self.last_synchronization.isoformat()
            }, f, indent=4)

        self.threshold_violations.save(path, csv_prefix='threshold_violations')
        pd.Series(self.horizon_updates).to_csv(os.path.join(path, 'horizon_updates.csv'), index=False)
        self.model_deployments.to_csv(os.path.join(path, 'deployments.csv'), index=True)

    @classmethod
    def load(cls, path: str, lazy_loading=False) -> 'Node':
        model = Model.load(path, lazy_loading=lazy_loading)
        lite_model = LiteModel.load(path, lazy_loading=lazy_loading)

        data = DataStorage.load(path)

        with open(os.path.join(path, 'node.json'), 'r') as f:
            d = json.load(f)
        n = cls(d['uuid'], ThresholdMetric.from_dict(d['threshold_metric']), model, Predictor(lite_model, data))
        n.last_synchronization = datetime.fromisoformat(d['last_synchronization'])

        n.threshold_violations = DataStorage.load(path, csv_prefix='threshold_violations')
        n.horizon_updates = pd.read_csv(os.path.join(path, 'horizon_updates.csv'), index_col=0, parse_dates=True).index
        n.model_deployments = pd.read_csv(os.path.join(path, 'deployments.csv'), index_col=0, parse_dates=True)
        return n
