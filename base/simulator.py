import json
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Optional, List

from base.learning_strategy import LearningStrategy
from base.deployment_strategy import DeploymentStrategy
from base.model import Model
from base.node import Node
from base.node_manager import NodeManager
from common import DataStorage, progress_bar, ModelMetadata, preprocess_df
from common.threshold_metric import ThresholdMetric


class SimulatorStrategy:
    """A small wrapper class for the simulator for a fixed strategy combination."""

    def __init__(self,
                 cl_strategy: LearningStrategy,
                 deploy_strategy: DeploymentStrategy
                 ) -> None:
        self.cl_strategy = cl_strategy
        self.deploy_strategy = deploy_strategy

    def __str__(self) -> str:
        return self.to_dict().__str__()

    def copy(self) -> 'SimulatorStrategy':
        return SimulatorStrategy(self.cl_strategy.copy(), self.deploy_strategy.copy())

    def to_dict(self) -> dict:
        return {'cl_strategy': self.cl_strategy.to_dict(), 'deploy_strategy': self.deploy_strategy.to_dict()}

    @classmethod
    def from_dict(cls, strategy: dict) -> 'SimulatorStrategy':
        return cls(LearningStrategy.from_dict(strategy['cl_strategy']),
                   DeploymentStrategy.from_dict(strategy['deploy_strategy'])
                   )


class SimulatorResult:
    """Data Wrapper for the high-level results of a simulation, including functions for analysis."""

    FILE_NAME = 'result.json'

    def __init__(self,
                 node_manager: NodeManager,
                 initial_df: Optional[pd.DataFrame],
                 continual_df: Optional[pd.DataFrame],
                 steps: int,
                 step_size: int,
                 base_model: ModelMetadata,
                 ) -> None:
        self.node_manager = node_manager
        self.initial_df = initial_df
        self.continual_df = continual_df
        self.steps = steps
        self.step_size = step_size
        self.base_model = base_model

    @property
    def node(self) -> Node:
        return self.node_manager.get_node(Simulator.NODE_ID)

    @property
    def data(self) -> DataStorage:
        return self.node.data

    @property
    def threshold_violations(self) -> DataStorage:
        return self.node_manager.get_threshold_violations(Simulator.NODE_ID)

    @property
    def num_threshold_violations(self) -> int:
        return len(self.node_manager.get_threshold_violations(Simulator.NODE_ID).get_measurements())

    @property
    def deployments(self) -> pd.DataFrame:
        return self.node_manager.get_model_deployments(Simulator.NODE_ID)

    @property
    def num_deployments(self) -> int:
        return len(self.node_manager.get_model_deployments(Simulator.NODE_ID))

    def horizon_updates(self) -> pd.DatetimeIndex:
        return self.node_manager.get_horizon_updates(Simulator.NODE_ID)

    @property
    def num_horizon_updates(self) -> int:
        return len(self.node_manager.get_horizon_updates(Simulator.NODE_ID))

    @property
    def message_exchanges(self) -> int:
        return self.num_threshold_violations + self.num_deployments + self.num_horizon_updates

    @property
    def mae(self) -> pd.Series:
        return self.node.data.mae

    @property
    def mse(self) -> pd.Series:
        return self.node.data.mse

    @property
    def rmse(self) -> pd.Series:
        return self.node.data.rmse

    @staticmethod
    def estimate_data_transferred_naive(steps: int, single_measurement_size: int) -> int:
        """Estimates the data transferred in bytes for the naive approach."""
        return steps * single_measurement_size

    def estimate_data_transferred(self,
                                  packet_overhead: int,
                                  measurement_size: int,
                                  horizon_length: int,
                                  ) -> int:
        """Estimates the total amount of data transferred in bytes."""
        tvs_data_points = (self.compute_time_until_threshold_violations() / timedelta(hours=1)).sum()
        return tvs_data_points * measurement_size \
            + self.num_threshold_violations * packet_overhead \
            + self.num_horizon_updates * (packet_overhead + horizon_length * measurement_size) \
            + self.deployments['size'].sum() \
            + self.num_deployments * packet_overhead

    def compute_time_until_threshold_violations(self) -> pd.Series:
        """Computes the duration from the last event (violation or horizon update) for every threshold violation."""
        dts = self.node.threshold_violations.get_measurements().index
        tvs = pd.DataFrame(index=dts, columns=['duration'], dtype='timedelta64[m]')
        dts = dts.to_pydatetime()

        if self.continual_df is None:
            logging.warning('No continual data available, assuming simulation started at first threshold violation.')
            sim_start = dts[0]
        else:
            sim_start = self.continual_df.index[0].to_pydatetime()

        if len(self.node.horizon_updates) > 0:
            hus = self.node.horizon_updates.to_pydatetime()
        else:
            hus = []

        dts = np.insert(dts, 0, [sim_start])
        for i, j in zip(dts[:-1], dts[1:]):
            last_action = i
            for dt in reversed(hus):
                if i < dt < j:
                    last_action = dt
                    break
            tvs.loc[j] = (j - last_action)
        return tvs['duration']

    def save(self, path='.', store_data: bool = False) -> None:
        try:
            os.makedirs(path, exist_ok=False)
        except FileExistsError:
            logging.warning(f'Path {path} already exists. Overwriting existing files.')

        # store all object data, but often we do not need to duplicate the simulation data
        if store_data:
            self.initial_df.to_csv(os.path.join(path, 'initial_data.csv'), index=True)
            self.continual_df.to_csv(os.path.join(path, 'continual_data.csv'), index=True)
        nm = self.node_manager
        nm.save(path)

        # store all configuration data and some metrics
        total_hours = self.steps / (3600 / self.step_size)
        result = {
            'steps': self.steps,
            'step_size': self.step_size,
            'base_model': self.base_model.to_dict(),
            'deployment_strategy': nm.deploy_strategy.to_dict(),
            'cl_strategy': nm.cl_strategy.to_dict(),
            'nodes': [],
        }

        node_id = Simulator.NODE_ID
        num_violations = self.num_threshold_violations
        result['nodes'].append(
            {
                'node_id': node_id,
                'model_parameters': nm.get_cl_model(node_id).get_parameter_count(),
                'threshold_metric': nm.get_node(node_id).threshold_metric.to_dict(),
                'threshold_violations': num_violations,
                'average_offset': nm.get_node(node_id).data.get_diff().mean().to_dict(),
                'steps_per_threshold_violation': self.steps / num_violations if num_violations > 0 else 0,
                'horizon_updates': self.num_horizon_updates,
                'optimal_horizon_updates': total_hours / self.base_model.output_length,
                'deployments': self.num_deployments,
                'average_deployment_size[B]': nm.get_model_deployments(node_id)['size'].mean(),
            }
        )

        with open(os.path.join(path, SimulatorResult.FILE_NAME), 'w') as f:
            json.dump(result, f, indent=4)

    @classmethod
    def load(cls, path='.', lazy_loading=False) -> 'SimulatorResult':
        # check if simulation data exists
        if not os.path.exists(os.path.join(path, 'initial_data.csv')) \
                or not os.path.exists(os.path.join(path, 'continual_data.csv')):
            print('No simulation data found. Loading only configuration data.')
            initial_df, continual_df = None, None
        else:
            initial_df = pd.read_csv(os.path.join(path, 'initial_data.csv'), index_col=0, parse_dates=True)
            continual_df = pd.read_csv(os.path.join(path, 'continual_data.csv'), index_col=0, parse_dates=True)

        # load configuration data
        nm = NodeManager.load(path=path, lazy_loading=lazy_loading)
        with open(os.path.join(path, SimulatorResult.FILE_NAME), 'r') as f:
            data = json.load(f)

        return cls(nm,
                   initial_df,
                   continual_df,
                   data['steps'],
                   data['step_size'],
                   ModelMetadata.from_dict(data['base_model'])
                   )

    def describe(self) -> None:
        """Prints a summary of the simulation results."""
        total_hours = self.steps / (3600 / self.step_size)
        nm = self.node_manager

        print(f'>>>>>>>>>>>>>>>> Simulation Results <<<<<<<<<<<<<<<<')
        print(f'Total Steps: {self.steps}')
        print(f'Step Size: {self.step_size} (= {total_hours / 24} days)\n')
        print(f'Deployment Strategy: {nm.deploy_strategy}')
        print(f'CL Strategy: {nm.cl_strategy}\n')

        for node_id in nm.node_ids:
            n = nm.get_node(node_id)
            print(f'-> NODE {node_id}')
            print(f'Threshold Metric: {n.threshold_metric}')

            violations = nm.get_threshold_violations(node_id)
            num_violations = len(violations.get_measurements())
            print(f'Threshold Violations: {num_violations}')
            if num_violations > 0:
                print(f'Steps per Threshold Violation: {self.steps / num_violations}')
            print(f'Average Offset: {nm.get_node(node_id).data.get_diff().mean().to_dict()}\n'),

            num_horizon_updates = len(nm.get_horizon_updates(node_id))
            print(f'Horizon Updates: {num_horizon_updates}')
            print(f'Optimal Horizon Updates: {total_hours / self.base_model.output_length}')

            deployments = nm.get_model_deployments(node_id)
            print(f'Deployments: {len(deployments)}')
            print(f'Average Deployment Size: {deployments["size"].mean()} Bytes')
            print(f'Latest Model Parameters: {nm.get_cl_model(node_id).get_parameter_count()}')
            print(f'Latest Deployed Model Size: {deployments["size"].iloc[-1]} Bytes')


class Simulator:
    NODE_ID = 'SIM'

    def __init__(self,
                 initial_df: pd.DataFrame,
                 continual_df: pd.DataFrame,
                 base_model_path: str,
                 model_dir: str,
                 strategy: SimulatorStrategy,
                 threshold_metric: ThresholdMetric,
                 resolution_in_seconds: int,
                 ) -> None:
        """
        Creates a new simulation object that can be used to run and inspect the results of multiple strategies at once.

        Args:
            initial_df: a DataFrame representing the data on which the first model was trained
            continual_df: the DataFrame on which to base the simulation for future measurements
            base_model_path: the path to the first model in the simulation
            model_dir: the directory that will be used to store all resulting models
            strategy: strategy to be used in the simulation
        """
        self.model_dir = model_dir
        self._base_model = Model.load(base_model_path)
        self.input_features = self._base_model.metadata.input_features
        self.output_features = self._base_model.metadata.output_features
        self.strategy = strategy
        self.threshold_metric = threshold_metric
        self.resolution_in_seconds = resolution_in_seconds
        self.node_manager: Optional[NodeManager] = None

        self._continual_df = preprocess_df(continual_df, self.input_features, [])  # no time series encoding
        self._initial_df = preprocess_df(initial_df, self.input_features, self._base_model.metadata.periodicity)

    def _init_deployment(self, start: datetime, initial_df: pd.DataFrame, lite_model=True):
        logging.info('Initializing simulation ...')

        hourly_df = initial_df[self.input_features].asfreq('H')

        cloned_model = self._base_model.clone()
        cloned_model.model.compile(optimizer=tf.optimizers.Adam(),
                                   loss=tf.losses.MeanSquaredError(),
                                   metrics=[tf.metrics.MeanAbsoluteError(),
                                            tf.metrics.RootMeanSquaredError()]
                                   )
        self.strategy.cl_strategy.add_model(cloned_model)

        nm = NodeManager(self.strategy.cl_strategy, self.strategy.deploy_strategy, self.model_dir)
        nm.add_node(Simulator.NODE_ID, self.threshold_metric, hourly_df, start, lite_model=lite_model)
        nm.on_model_deployment(Simulator.NODE_ID, start)
        self.node_manager = nm

    def run(self, path: str, lite_model: bool = True, store_data: bool = False, verbose: int = 1):
        """Runs the simulation and stores the results in the given path."""
        start = self._continual_df.index.min().to_pydatetime()
        end = self._continual_df.index.max().to_pydatetime()
        self._init_deployment(start, self._initial_df, lite_model=lite_model)

        logging.info(f'Starting deployment strategies simulation "{path}"...')
        # upsample the actual measurements for the required resolution (linear interpolation)
        iterate_df: pd.DataFrame = self._continual_df \
            .resample(timedelta(seconds=self.resolution_in_seconds)) \
            .interpolate(method='time')

        hourly_df: pd.DataFrame = self._continual_df.asfreq('H')

        nm = self.node_manager
        last_sync = self._initial_df.index.max().to_pydatetime()
        prediction_buffer = []
        for values in progress_bar(iterate_df, prefix=f'Simulating from {start} to {end}', verbose=verbose):
            step_dt = values[0].to_pydatetime()

            try:
                prediction = nm.get_prediction_at(Simulator.NODE_ID, step_dt)
            except ValueError:
                logging.debug(f'Reached prediction horizon at {step_dt}')
                new_measurements = hourly_df.loc[last_sync:step_dt]
                last_sync = step_dt + timedelta(seconds=self.resolution_in_seconds)  # mitigate redundant values
                if nm.on_horizon_update(Simulator.NODE_ID, step_dt, new_measurements) is not None:
                    nm.on_model_deployment(Simulator.NODE_ID, step_dt)
                prediction = nm.get_prediction_at(Simulator.NODE_ID, step_dt)

            measurement = np.asarray(values[1:])
            if self.threshold_metric.is_threshold_violation(measurement, prediction.to_numpy()):
                logging.debug(f'Threshold violation at {step_dt}')
                new_measurements = hourly_df.loc[last_sync:step_dt]
                last_sync = step_dt + timedelta(seconds=self.resolution_in_seconds)  # mitigate redundant values
                if nm.on_threshold_violation(Simulator.NODE_ID, step_dt, measurement, new_measurements) is not None:
                    nm.on_model_deployment(Simulator.NODE_ID, step_dt)

            prediction_buffer.append(prediction)

        # needed for analyzing metrics over the entire simulation
        nm.get_node(Simulator.NODE_ID).data._measurements = iterate_df
        nm.get_node(Simulator.NODE_ID).data._predictions = pd.DataFrame(prediction_buffer)

        logging.info(f'Storing results to {path}...\n')
        result = SimulatorResult(
            nm,
            self._initial_df,
            iterate_df,
            len(iterate_df),
            self.resolution_in_seconds,
            self._base_model.metadata
        )
        result.save(path, store_data=store_data)

        return result

    @staticmethod
    def analyze_simulation_result(path: str):
        """Print an overview of the simulation results in the given directory."""
        print(f'Loading simulation results in {path}...')
        SimulatorResult.load(path).describe()

    @staticmethod
    def run_delta_simulation(initial_df: pd.DataFrame,
                             continual_df: pd.DataFrame,
                             threshold_metric: ThresholdMetric,
                             resolution_in_seconds: int,
                             input_features: List[str],
                             output_features: List[str],
                             path: str,
                             verbose: int = 1,
                             ):
        initial_df = preprocess_df(initial_df, input_features, [])
        continual_df = preprocess_df(continual_df, input_features, [])
        input_to_output_indices = [input_features.index(f) for f in output_features]

        iterate_df: pd.DataFrame = continual_df \
            .resample(timedelta(seconds=resolution_in_seconds)) \
            .interpolate(method='time')

        predictions: List[pd.Series] = []
        tv_measurements: List[tuple] = []  # pd.DataFrame.itertuples
        tv_predictions: List[pd.Series] = []
        prediction = initial_df.loc[initial_df.index.max(), output_features]
        for values in progress_bar(iterate_df[input_features], verbose=verbose):
            step_dt = values[0].to_pydatetime()
            prediction.name = step_dt
            predictions.append(prediction.copy())

            measurement = np.asarray(values[1:])
            if threshold_metric.is_threshold_violation(measurement, prediction.to_numpy()):
                logging.debug(f'Threshold violation at {step_dt}')
                logging.debug(f'New prediction: {values}')
                tv_measurements.append(values)
                prediction = pd.Series(data=measurement[input_to_output_indices],
                                       index=output_features,
                                       name=step_dt
                                       )
                tv_predictions.append(prediction.copy())

        # store predictions, measurements and threshold violations
        node_path = os.path.join(path, Simulator.NODE_ID)

        data = DataStorage.from_data(iterate_df, pd.DataFrame(predictions))
        data.save(node_path)

        tv_measurements: pd.DataFrame = pd.DataFrame(tv_measurements)
        tv_measurements.set_index('Index', inplace=True)
        tvs = DataStorage.from_data(tv_measurements, pd.DataFrame(tv_predictions))
        tvs.save(node_path, 'threshold_violations')
