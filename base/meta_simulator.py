import json
import logging
import os
import sys
from typing import List, Dict, Optional, Callable

import pandas as pd
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Repeat(tf.keras.layers.Layer):
    def __init__(self, output_length: int, output_indices: List[int], **kwargs):
        super(Repeat, self).__init__(**kwargs)
        self.output_length = output_length
        self.output_indices = output_indices

    def call(self, inputs, **kwargs):
        last_values = tf.stack([inputs[:, -1, i] for i in self.output_indices], axis=-1)
        one_day_ago = tf.stack([inputs[:, -25, i] for i in self.output_indices], axis=-1)
        diff = one_day_ago - last_values
        diff = tf.expand_dims(diff, axis=-1)
        diff = tf.repeat(diff, repeats=self.output_length, axis=1)

        outputs = tf.stack([inputs[:, -self.output_length:, i] for i in self.output_indices], axis=-1)
        return outputs - diff

    def get_config(self):
        config = super(Repeat, self).get_config()
        config.update({
            'output_length': self.output_length,
            'output_indices': self.output_indices
            }
            )
        return config


class MetaSimulator:
    """Can run multiple `Simulator` instances at once."""

    FILE_NAME = 'simulation.json'

    def __init__(self,
                 simulator_data: dict,
                 strides: List[int],
                 threshold_metrics: Dict[str, dict],
                 strategies: Dict[str, dict],
                 model_dir: str,
                 base_model_id: str,
                 result_dir: str = '.',
                 ) -> None:
        self.simulator_data = simulator_data
        self.strides = strides
        self.threshold_metrics = threshold_metrics
        self.strategies = strategies
        self.model_dir = model_dir
        self.base_model_id = base_model_id
        self.result_dir = result_dir

    def __repr__(self) -> str:
        return f'MetaSimulator({self.to_dict()})'

    def to_dict(self) -> dict:
        return {
            'simulator_data': self.simulator_data,
            'strides': self.strides,
            'threshold_metrics': self.threshold_metrics,
            'strategies': self.strategies,
            'model_dir': self.model_dir,
            'base_model_id': self.base_model_id,
            'result_dir': self.result_dir,
            }

    def save(self, path: str = '.') -> None:
        logging.info(f'Saving meta simulation to {path}')
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'MetaSimulator':
        if not os.path.isdir(path):
            raise ValueError('Path is not a directory')
        else:
            logging.info(f'Loading meta simulation from directory {path}')

        with open(os.path.join(path, MetaSimulator.FILE_NAME), 'r') as f:
            data = json.load(f)

        with os.scandir(path) as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_dir():
                    stride = int(entry.name[:-1])  # 600s -> 600
                    if stride not in data['strides']:
                        print(f'Found directory {entry.name} but no stride {stride} in meta simulation')
                        data['strides'].append(stride)

        for stride in data['strides']:
            with os.scandir(os.path.join(path, f'{stride}s')) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_dir():
                        metric = entry.name
                        if metric not in data['threshold_metrics']:
                            print(f'Found directory {entry.name} but no metric {metric} in meta simulation')
                            data['threshold_metrics'][metric] = {}

            for metric in data['threshold_metrics']:
                with os.scandir(os.path.join(path, f'{stride}s', metric)) as it:
                    for entry in it:
                        if not entry.name.startswith('.') and entry.is_dir():
                            strategy = entry.name
                            if strategy not in data['strategies']:
                                print(f'Found directory {entry.name} but no strategy {strategy} in meta simulation')
                                data['strategies'][strategy] = {}
        return cls(**data)

    def run_simulations(self,
                        verbose: int = 1,
                        store_data: bool = False,
                        callback: Optional[Callable] = None,
                        ) -> None:
        base_model_path = os.path.join(self.model_dir, self.base_model_id)
        for stride in self.strides:
            for metric_name, metric in self.threshold_metrics.items():
                for strategy_name, strategy in self.strategies.items():
                    logging.info(f'Starting simulation with '
                                 f'stride {stride}s, metric {metric_name}, and {strategy_name} strategy'
                                 )
                    MetaSimulator.run_single_sim(self.simulator_data,
                                                 base_model_path,
                                                 self.model_dir,
                                                 stride,
                                                 strategy,
                                                 strategy_name,
                                                 metric,
                                                 metric_name,
                                                 self.result_dir,
                                                 store_data=store_data,
                                                 verbose=verbose,
                                                 )
                    if callback is not None:
                        callback()
        self.save(os.path.join(self.result_dir, MetaSimulator.FILE_NAME))
        print('Finished simulations')

    def run_delta_simulations(self, verbose: int = 1):
        base_model_path = os.path.join(self.model_dir, self.base_model_id)
        for stride in self.strides:
            for metric_name, metric in self.threshold_metrics.items():
                print(f'Starting simulation with stride {stride}s, metric {metric_name}, and delta strategy')
                MetaSimulator.run_single_delta_sim(self.simulator_data, base_model_path, stride,
                                                   metric, metric_name, self.result_dir, verbose=verbose,
                                                   )
        print('Finished delta simulations')

    def run_repeat_simulations(self, verbose: int = 1):
        base_model_path = os.path.join(self.model_dir, self.base_model_id)

        for stride in self.strides:
            for metric_name, metric in self.threshold_metrics.items():
                logging.info(f'Starting simulation with '
                             f'stride {stride}s, metric {metric_name} and repeat strategy'
                             )
                MetaSimulator.run_single_repeat_sim(self.simulator_data,
                                                    base_model_path,
                                                    self.model_dir,
                                                    stride,
                                                    metric,
                                                    metric_name,
                                                    self.result_dir,
                                                    verbose=verbose,
                                                    )

    def run_simulations_multiprocessing(self,
                                        initial_df: pd.DataFrame,
                                        continual_df: pd.DataFrame,
                                        base_model_id: str,
                                        model_dir: str,
                                        result_dir: str = '.',
                                        ) -> None:
        # TODO: multiprocessing with tensorflow is not working
        # challenging to use multiprocessing with tensorflow, c.f.
        # https://stackoverflow.com/questions/36610290/tensorflow-and-multiprocessing-passing-sessions
        import multiprocessing
        multiprocessing.set_start_method('spawn')
        # logger = multiprocessing.log_to_stderr()
        # logger.setLevel(multiprocessing.SUBDEBUG)

        # use CPU because GPU has problems with multiprocessing
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        base_model_path = os.path.join(model_dir, base_model_id)
        with multiprocessing.Pool(processes=12) as pool:
            for stride in self.strides:
                for metric_name, metric in self.threshold_metrics.items():
                    for strategy_name, strategy in self.strategies.items():
                        logging.info(f'Starting simulation with stride {stride}s, metric {metric_name}, '
                                     f'and {strategy_name} strategy'
                                     )
                        pool.apply_async(MetaSimulator.run_single_sim,
                                         args=(initial_df.copy(), continual_df.copy(), base_model_path,
                                               model_dir, stride, strategy, strategy_name,
                                               metric, metric_name, result_dir, True)
                                         )
            pool.close()
            pool.join()

    @staticmethod
    def run_single_sim(sim_data: dict,
                       base_model_path: str,
                       model_dir: str,
                       stride: int,
                       strategy: dict,
                       strategy_name: str,
                       metric: dict,
                       metric_name: str,
                       result_dir: str,
                       store_data: bool = False,
                       verbose: int = 1,
                       ) -> None:
        # we do the imports here so that tensorflow is loaded separately for every process
        import tensorflow as tf
        from base.simulator import Simulator, SimulatorStrategy, SimulatorResult
        from base.simulator_data import SimulatorData
        from common import ThresholdMetric

        sim_dir = os.path.join(result_dir, f'{stride}s', metric_name, strategy_name)
        result_dir = os.path.join(os.getcwd(), sim_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            # check if there are already results
            if os.path.exists(os.path.join(result_dir, SimulatorResult.FILE_NAME)):
                print(f'Skipping simulation because there is already a result in {result_dir}', file=sys.stderr)
                return
            print(f'No result file found in {result_dir}, but directory exists. Overwriting.', file=sys.stderr)

        logging.basicConfig(force=True,
                            level=logging.DEBUG,
                            filename=os.path.join(result_dir, 'stdout.txt'),
                            filemode='w'
                            )

        tf_log = tf.get_logger()
        fh = logging.FileHandler(os.path.join(result_dir, 'stdout.txt'), mode='a')
        tf_log.addHandler(fh)

        sim_data = SimulatorData.from_dict(sim_data)
        sim = Simulator(sim_data.initial_df,
                        sim_data.continual_df,
                        base_model_path,
                        os.path.join(model_dir, sim_dir),
                        SimulatorStrategy.from_dict(strategy),
                        ThresholdMetric.from_dict(metric),
                        stride
                        )
        sim.run(sim_dir, store_data=store_data, verbose=verbose)

    @staticmethod
    def run_single_delta_sim(sim_data: dict,
                             base_model_path: str,
                             stride: int,
                             metric: dict,
                             metric_name: str,
                             result_dir: str,
                             verbose: int = 1,
                             ) -> None:

        sim_dir = os.path.join(result_dir, f'{stride}s', metric_name, 'delta')
        result_dir = os.path.join(os.getcwd(), sim_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            print(f'Skipping simulation, directory already exists at {result_dir}', file=sys.stderr)
            return
        logging.basicConfig(force=True,
                            level=logging.DEBUG,
                            filename=os.path.join(result_dir, 'stdout.txt'),
                            filemode='w'
                            )

        from base.simulator import Simulator
        from base.simulator_data import SimulatorData
        from common import ThresholdMetric, ModelMetadata

        model_md = ModelMetadata.load(os.path.join(base_model_path, ModelMetadata.FILE_NAME))
        sim_data = SimulatorData.from_dict(sim_data)
        Simulator.run_delta_simulation(sim_data.initial_df,
                                       sim_data.continual_df,
                                       ThresholdMetric.from_dict(metric),
                                       stride,
                                       model_md.input_features,
                                       model_md.output_features,
                                       sim_dir,
                                       verbose=verbose,
                                       )

    @staticmethod
    def run_single_repeat_sim(sim_data: dict,
                              base_model_path: str,
                              model_dir: str,
                              stride: int,
                              metric: dict,
                              metric_name: str,
                              result_dir: str,
                              verbose: int = 1,
                              ) -> None:
        sim_dir = os.path.join(result_dir, f'{stride}s', metric_name, 'repeat')
        result_dir = os.path.join(os.getcwd(), sim_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            print(f'Skipping simulation, directory already exists at {result_dir}', file=sys.stderr)
            return
        logging.basicConfig(force=True,
                            level=logging.DEBUG,
                            filename=os.path.join(result_dir, 'stdout.txt'),
                            filemode='w',
                            )

        from base.simulator import Simulator, SimulatorStrategy
        from base.simulator_data import SimulatorData
        from base.model import Model

        from common import ThresholdMetric, ModelMetadata

        model_md = ModelMetadata.load(os.path.join(base_model_path, ModelMetadata.FILE_NAME))
        sim_data = SimulatorData.from_dict(sim_data)

        inputs = tf.keras.Input((model_md.input_length, model_md.input_attributes))
        repeat = Repeat(model_md.output_length, model_md.input_to_output_indices)
        outputs = repeat(inputs)
        repeat_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        repeat_model.compile(loss=tf.losses.MeanSquaredError(),
                             metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()],
                             )
        model_md.context = None
        model = Model(repeat_model, model_md)
        base_model_path = os.path.join(model_dir, 'repeat_baseline')
        model.save(base_model_path)

        static_strategy = {
            'cl_strategy': {'type': 'NoUpdateStrategy', 'object': {}},
            'deploy_strategy': {'type': 'DeployOnceStrategy', 'object': {}}}

        sim = Simulator(sim_data.initial_df,
                        sim_data.continual_df,
                        base_model_path,
                        os.path.join(model_dir, sim_dir),
                        SimulatorStrategy.from_dict(static_strategy),
                        ThresholdMetric.from_dict(metric),
                        stride,
                        )
        sim.run(sim_dir, verbose=verbose)
