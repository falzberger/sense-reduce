import logging
import re
from typing import Optional

import pandas as pd

from abstract_sensor import AbstractSensor


class TemperatureSensor(AbstractSensor):
    """Encapsulates the access to a temperature sensor."""

    path = '/sys/bus/w1/devices/28-3c01e076eb54/w1_slave'

    def __init__(self) -> None:
        super(TemperatureSensor, self).__init__()

    @property
    def measurement(self) -> pd.Series:
        return pd.Series(data=[self.temperature], index=['TMP'])

    @property
    def temperature(self) -> Optional[float]:
        """The current temperature in degrees Celsius or None if measurement failed."""
        temperature = None
        try:
            f = open(self.path, "r")
            line = f.readline()
            if re.match(r"([\da-f]{2} ){9}: crc=[\da-f]{2} YES", line):
                line = f.readline()
                m = re.match(r"([\da-f]{2} ){9}t=([+-]?\d+)", line)
                if m:
                    temperature = float(m.group(2)) / 1000.0
            f.close()
        except Exception as e:
            logging.error(f'Error reading {self.path}: {e}')
        return temperature


if __name__ == '__main__':
    import argparse
    import datetime
    import sys
    import time

    parser = argparse.ArgumentParser(description='Use DS18B20 to measure current temperature')
    parser.add_argument('--output', type=str,
                        help='The path of the output file, if none given, measurements are printed to console.')
    args = parser.parse_args()

    output = open(args.output, 'wt') if args.output is not None else sys.stdout
    sensor = TemperatureSensor()
    print('Datetime, Temperature (°C)', file=output)
    while True:
        temp = sensor.temperature
        if temp is None:
            continue
        print(f'{datetime.datetime.now().isoformat()},{temp}', file=output)
        output.flush()
        time.sleep(2)
