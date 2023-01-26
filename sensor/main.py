import datetime
import functools
import logging
import os
import time
import uuid

import numpy as np
import pandas as pd
import requests

from common import Predictor, LiteModel, DataStorage, ThresholdMetric, ModelMetadata, L2Threshold
from predicting_monitor import PredictingMonitor

logging.basicConfig(level=logging.DEBUG)


def run(threshold_metric: ThresholdMetric, base: str, mode: str, wifi_toggle: bool, interval_seconds: float):
    """Starts running a sensor node in SenseReduce that will register at the base station and start monitoring.

    Args:
        threshold_metric: The defined threshold metric for the sensor node.
        base: The address of the base station, e.g., 192.168.0.1:100.
        mode: The mode applied for data reduction, either 'none' or 'predict'.
        wifi_toggle: A flag whether Wi-Fi should be turned off in-between transmissions.
        interval_seconds: Defines the regular checking interval in seconds.
    """
    logging.info(
        f'Starting sensor node with ID={NODE_ID} in "{mode}" mode, threshold={threshold_metric} and base={base}...'
    )
    sensor = TemperatureSensor()

    if mode == 'none':
        register_node(base, threshold_metric)
        while True:
            now = datetime.datetime.now()
            send_measurement(now, sensor.measurement.values, base)
            time.sleep(interval_seconds)

    elif mode == 'predict':
        model, data = fetch_model_and_data(base, threshold_metric)
        predictor = Predictor(model, data)
        predictor.update_prediction_horizon(datetime.datetime.now())
        monitor = PredictingMonitor(sensor, predictor)

        monitor.monitor(threshold_metric=threshold_metric,
                        interval_seconds=interval_seconds,
                        violation_callback=functools.partial(wifi_wrapper, wifi_toggle, send_violation, base=base),
                        update_callback=functools.partial(wifi_wrapper, wifi_toggle, send_update, base=base)
                        )

    else:
        logging.error(f'Unsupported data reduction mode: {mode}')
        exit(1)


def register_node(base: str, threshold_metric: ThresholdMetric) -> requests.Response:
    """Registers the node at the base station by informing it about the node id and the threshold metric.

    Returns:
        The request's response, containing model metadata and initial data in the body.
    """
    body = {
        'threshold_metric': threshold_metric.to_dict()
    }
    logging.debug(f'Node {NODE_ID} registering with: {body}')
    return requests.post(f'{base}/register/{NODE_ID}', json=body)


def fetch_model_and_data(base: str, threshold_metric: ThresholdMetric) -> (LiteModel, DataStorage):
    body = register_node(base, threshold_metric).json()

    metadata = ModelMetadata.from_dict(body.get('model_metadata'))
    r = requests.get(f'{base}/models/{NODE_ID}')
    file_name = f'{metadata.uuid}.tflite'
    open(file_name, 'wb').write(r.content)

    initial_df = pd.read_json(body.get('initial_df'))
    logging.debug(f'Node {NODE_ID} fetched initial data for prediction model: {initial_df}')
    data = DataStorage(metadata.input_features, metadata.output_features)
    data.add_measurement_df(initial_df)

    return LiteModel.from_tflite_file(file_name, metadata), data


def wifi_wrapper(wifi_toggle: bool, func, *args, **kwargs):
    if wifi_toggle:
        os.system('sudo rfkill unblock wifi')
        # TODO: improve busy waiting of Wi-Fi toggling
        # TODO: use a more reliable way to check if Wi-Fi is connected
        before = time.time_ns()
        busy_waiting = True
        while busy_waiting:
            try:
                requests.get('http://192.168.8.110:5000/ping')
                busy_waiting = False
            except:
                pass
        print(f'Took {time.time_ns() - before} ns to establish connection')
    func(*args, **kwargs)
    if wifi_toggle:
        os.system('sudo rfkill block wifi')


def send_measurement(dt: datetime.datetime, measurement: np.ndarray, base: str):
    body = {
        'timestamp': dt.isoformat(),
        'measurement': list(measurement),
    }
    logging.debug(f'Node {NODE_ID} sending measurement: {body}')
    requests.post(f'{base}/measurement/{NODE_ID}', json=body)


def send_update(dt: datetime.datetime, data: pd.DataFrame, base: str):
    body = {
        'timestamp': dt.isoformat(),
        'data': data.to_json(),
    }
    logging.debug(f'Node {NODE_ID} sending update: {body}')
    requests.post(f'{base}/update/{NODE_ID}', json=body)


def send_violation(dt: datetime.datetime, measurement: np.ndarray, data: pd.DataFrame, base: str):
    body = {
        'timestamp': dt.isoformat(),
        'measurement': list(measurement),
        'data': data.to_json(),
    }
    logging.debug(f'Node {NODE_ID} handling violation by sending: {body}')
    requests.post(f'{base}/violation/{NODE_ID}', json=body)
    # TODO: handle response that may contain a new model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start a new sensor node.')
    parser.add_argument('sensor', type=str, choices=['ds18b20', 'sense-hat', 'dht22', 'mock'],
                        help='The sensor type to use for measurements.'
                        )
    parser.add_argument('base', type=str,
                        help='The address of the base station, e.g., 192.168.0.1:100.'
                        )
    parser.add_argument('--mode', type=str, choices=['none', 'predict'], default='predict',
                        help='The operation mode for data reduction, either "none" or "predict", default: "predict"'
                        )
    parser.add_argument('--wifi', action='store_true',
                        help='A flag for turning off Wi-Fi in-between transmissions.'
                        )
    parser.add_argument('--interval', type=float, default=5.0,
                        help='The regular monitoring interval for measurements in seconds.'
                        )
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='The threshold in degrees Celsius above which to report to the base station, default: 1.0.'
                        )
    parser.add_argument('--id', type=str, default=uuid.uuid1(),
                        help='The unique ID of the node. If multiple nodes have the same ID, behavior is undefined.'
                             'If not provided, a UUID is generated with uuid.uuid1().'
                        )
    ARGS = parser.parse_args()

    if ARGS.sensor == 'ds18b20':
        from temperature_sensor_ds18b20 import TemperatureSensor
    elif ARGS.sensor == 'sense-hat':
        from temperature_sensor_sense_hat import TemperatureSensor
    elif ARGS.sensor == 'dht22':
        from temperature_sensor_dht22 import TemperatureSensor
    elif ARGS.sensor == 'mock':
        from temperature_sensor_mock import TemperatureSensor
    else:
        logging.error(f'Unsupported sensor type: {ARGS.sensor}. Aborting...')
        exit(1)

    NODE_ID = ARGS.id
    THRESHOLD = L2Threshold(ARGS.threshold, [0], [0])
    run(THRESHOLD, ARGS.base, ARGS.mode, ARGS.wifi, ARGS.interval)
