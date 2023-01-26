import logging
from typing import Optional

import pandas as pd
from sense_hat import SenseHat

from abstract_sensor import AbstractSensor


class TemperatureSensor(AbstractSensor):
    """Encapsulates the access to a temperature sensor."""

    def __init__(self) -> None:
        self._sensor = SenseHat()

    @property
    def measurement(self) -> pd.Series:
        return pd.Series(data=[self.temperature], index=['TMP'])

    @property
    def temperature(self) -> Optional[float]:
        """The current temperature in degrees Celsius or None if measurement failed."""
        try:
            return self._sensor.temperature
        except RuntimeError as e:
            logging.error(e.args[0])
            return None

    @property
    def humidity(self) -> Optional[float]:
        """The current humidity in percent or None if measurement failed."""
        try:
            return self._sensor.humidity
        except RuntimeError as e:
            logging.error(e.args[0])
            return None


if __name__ == '__main__':
    import argparse
    import datetime
    import sys
    import time

    parser = argparse.ArgumentParser(description='Use SenseHat to measure temperature and humidity.')
    parser.add_argument('--output', type=str,
                        help='The path of the output file, if none given, measurements are printed to console.')
    args = parser.parse_args()

    output = open(args.output, 'wt') if args.output is not None else sys.stdout
    sensor = TemperatureSensor()
    print('Datetime, Temperature (°C), Humidity (%)', file=output)
    while True:
        temp = sensor.temperature
        hum = sensor.humidity
        if temp is None or hum is None:
            continue
        print(f'{datetime.datetime.now().isoformat()},{temp},{hum}', file=output)
        output.flush()
        time.sleep(2)
