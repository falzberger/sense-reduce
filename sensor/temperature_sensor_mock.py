import random

import pandas as pd

from abstract_sensor import AbstractSensor


class TemperatureSensor(AbstractSensor):
    """Mocks a temperature sensor with random data."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def measurement(self) -> pd.Series:
        return pd.Series(data=[self.temperature], index=['TMP'])

    @property
    def temperature(self) -> float:
        return random.random() * 10 + 10  # interval [10, 20]

    @property
    def humidity(self) -> float:
        return random.random() * 30 + 60  # interval [60, 90]
