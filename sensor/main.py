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


def run(threshold_metric: ThresholdMetric,
        base_address: str,
        data_reduction_mode: str,
        wifi_toggle: bool,
        check_interval: float,
        ) -> None:
    """Starts a sensor node in SenseReduce, which connects to a base station and starts monitoring.

    Args:
        threshold_metric (ThresholdMetric): The metric used to determine if a threshold has been reached.
        base_address (str): The address of the base station, e.g., "192.168.0.1:100".
        data_reduction_mode (str): The data reduction mode applied, either "none" or "predict".
        wifi_toggle (bool): A flag indicating whether Wi-Fi should be turned off between transmissions.
        check_interval (float): The regular interval in seconds for checking the sensor's readings.

    Returns:
        None

    Raises:
        ValueError: If an invalid data reduction mode is specified.
    """

    logging.info(f'Starting sensor node with ID={NODE_ID} in "{data_reduction_mode}" mode, '
                 f'threshold={threshold_metric} and base={base_address}...'
                 )
    sensor = TemperatureSensor()

    if data_reduction_mode == 'none':
        register_node(base_address, threshold_metric)
        while True:
            current_time = datetime.datetime.now()
            send_measurement(current_time, sensor.measurement.values, base_address)
            time.sleep(check_interval)

    elif data_reduction_mode == 'predict':
        model, data = fetch_model_and_data(base_address, threshold_metric)
        predictor = Predictor(model, data)
        predictor.update_prediction_horizon(datetime.datetime.now())
        monitor = PredictingMonitor(sensor, predictor)

        monitor.monitor(threshold_metric=threshold_metric,
                        interval_seconds=check_interval,
                        violation_callback=functools.partial(wifi_wrapper,
                                                             wifi_toggle,
                                                             send_violation,
                                                             base_address=base_address,
                                                             monitor=monitor,
                                                             ),
                        update_callback=functools.partial(wifi_wrapper,
                                                          wifi_toggle,
                                                          send_update,
                                                          base_address=base_address,
                                                          monitor=monitor,
                                                          )
                        )

    else:
        raise ValueError(f'Unsupported data reduction mode: {data_reduction_mode}')


def register_node(base_address: str, threshold_metric: ThresholdMetric) -> requests.Response:
    """Registers the sensor node with the base station by providing its ID and threshold metric.

    Args:
        base_address: The address of the base station, e.g., "192.168.0.1:100".
        threshold_metric: The metric used to determine if a threshold has been reached.

    Returns:
        requests.Response: The response from the base station containing metadata and initial data in the body.

    Raises:
        requests.exceptions.RequestException: If an error occurs while sending the request.
    """
    body = {'threshold_metric': threshold_metric.to_dict()}
    logging.debug(f'Registering node {NODE_ID} with base station at {base_address}: {body}')
    try:
        response = requests.post(f'{base_address}/register/{NODE_ID}', json=body)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f'Registration failed: {e}')
        raise


def fetch_model_and_data(base_address: str, threshold_metric: ThresholdMetric) -> (LiteModel, DataStorage):
    """Fetches the prediction model and initial data from the base station.

    Args:
        base_address: The address of the base station, e.g., "192.168.0.1:100".
        threshold_metric: The metric used to determine if a threshold has been reached.

    Returns:
        A tuple with the prediction model loaded from a TensorFlow Lite file and the initial data used to train it.

    Raises:
        requests.exceptions.RequestException: If an error occurs while sending the request or loading the model.
    """
    try:
        # Register the node to receive model metadata and initial data
        response = register_node(base_address, threshold_metric)
        body = response.json()

        # Download the model file from the base station and load it into a LiteModel object
        metadata = ModelMetadata.from_dict(body.get('model_metadata'))
        model = fetch_model(base_address, metadata)

        # Load the initial data into a DataStorage object
        initial_df = pd.read_json(body.get('initial_df'))
        logging.debug(f'Node {NODE_ID} fetched initial data for prediction model: {initial_df}')
        data = DataStorage(metadata.input_features, metadata.output_features)
        data.add_measurement_df(initial_df)

        return model, data

    except (requests.exceptions.RequestException, ValueError) as e:
        logging.error(f'Failed to fetch prediction model and data from base station: {e}')
        raise


def fetch_model(base_address: str, model_metadata: ModelMetadata) -> LiteModel:
    """Fetches the prediction model from the base station.

    Args:
        base_address: The address of the base station, e.g., "192.168.0.1:100".
        model_metadata: The metadata of the model to fetch.

    Returns:
        The prediction model loaded from a TensorFlow Lite file.

    Raises:
        requests.exceptions.RequestException: If an error occurs while sending the request or loading the model.
    """
    try:
        r = requests.get(f'{base_address}/models/{NODE_ID}')
        file_name = f'{model_metadata.uuid}.tflite'
        open(file_name, 'wb').write(r.content)
        return LiteModel.from_tflite_file(file_name, model_metadata)
    except requests.exceptions.RequestException as e:
        logging.error(f'Failed to fetch prediction model from base station {base_address}: {e}')
        raise


def wifi_wrapper(wifi_toggle: bool, func, *args, **kwargs):
    """
    Wrapper function that handles toggling Wi-Fi before and after a function call.
    If `wifi_toggle` is True, Wi-Fi is turned on before the function call and off after it.
    The function call is executed regardless of the `wifi_toggle` setting.
    """
    if wifi_toggle:
        os.system('sudo rfkill unblock wifi')
        # wait for Wi-Fi to connect
        base_url = kwargs.get('base', 'http://192.168.8.110:5000')
        wait_for_wifi(base_url)
    func(*args, **kwargs)
    if wifi_toggle:
        os.system('sudo rfkill block wifi')


def wait_for_wifi(base_url: str, timeout: int = 30):
    """
    Waits for the Wi-Fi to connect to the base station.
    """
    start_time = time.monotonic()
    while True:
        try:
            r = requests.get(f'{base_url}/ping')
            if r.ok:
                end_time = time.monotonic()
                logging.debug(f'Connected to {base_url} in {end_time - start_time} seconds')
                return
        except requests.exceptions.RequestException:
            pass
        elapsed_time = time.monotonic() - start_time
        if elapsed_time >= timeout:
            logging.warning(f'Timed out waiting for Wi-Fi to connect to {base_url}')
            return
        time.sleep(0.5)  # wait 0.5 second before retrying


def send_measurement(dt: datetime.datetime, measurement: np.ndarray, base_address: str):
    """Sends a single measurement to the specified base address.

    Args:
        dt: The measurement's timestamp as a datetime object.
        measurement: The measurement as a NumPy array.
        base_address: The address of the base station to send the measurement to.

    Raises:
        requests.exceptions.RequestException: An error occurred while sending the measurement.

    """
    body = {
        'timestamp': dt.isoformat(),
        'measurement': list(measurement),
    }
    logging.debug(f'Node {NODE_ID} sending measurement: {body}')
    try:
        response = requests.post(f'{base_address}/measurement/{NODE_ID}', json=body)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f'Error sending measurement: {e}')
        raise e


def send_update(dt: datetime.datetime, data: pd.DataFrame, base_address: str, monitor: PredictingMonitor):
    """Communicates a horizon update to the specified base address.

    Args:
        dt: The timestamp at which the horizon udpate is necessary a datetime object.
        data: The (redcued) measurements that occured in the current prediction horizon as a NumPy array.
        base_address: The address of the base station.
        monitor: The monitor that is used for prediction-based the reduction.

    Raises:
        requests.exceptions.RequestException: An error occurred while sending the horizon update.

    """
    body = {
        'timestamp': dt.isoformat(),
        'data': data.to_json(),
    }
    logging.debug(f'Node {NODE_ID} sending horizon update: {body}')

    try:
        response = requests.post(f'{base_address}/update/{NODE_ID}', json=body)
        response.raise_for_status()

        body = response.json()
        model_metadata = body.get('model_metadata')
        if model_metadata is not None:
            model_metadata = ModelMetadata.from_dict(model_metadata)
            model = fetch_model(base_address, model_metadata)

            new_predictor = Predictor(model, monitor.predictor.data)
            new_predictor.update_prediction_horizon(dt)
            monitor.predictor = new_predictor

    except requests.exceptions.RequestException as e:
        logging.error(f'Error sending horizon update: {e}')
        raise e


def send_violation(dt: datetime.datetime,
                   measurement: np.ndarray,
                   data: pd.DataFrame,
                   base_address: str,
                   monitor: PredictingMonitor
                   ):
    """
    Sends a violation message to the base station, containing the timestamp of the violation, the measurement that
    triggered it, and the data required for updating the prediction horizon.
    """
    body = {
        'timestamp': dt.isoformat(),
        'measurement': list(measurement),
        'data': data.to_json(),
    }
    logging.debug(f'Node {NODE_ID} handling violation by sending: {body}')

    try:
        response = requests.post(f'{base_address}/violation/{NODE_ID}', json=body)
        response.raise_for_status()

        body = response.json()

        model_metadata = body.get('model_metadata')
        if model_metadata is not None:
            model_metadata = ModelMetadata.from_dict(model_metadata)
            model = fetch_model(base_address, model_metadata)

            new_predictor = Predictor(model, monitor.predictor.data)
            new_predictor.update_prediction_horizon(dt)
            monitor.predictor = new_predictor

    except requests.exceptions.RequestException as e:
        logging.error(f'Error sending horizon update: {e}')
        raise e


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
