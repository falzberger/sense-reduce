from abc import ABC, abstractmethod

import pandas as pd


class AbstractSensor(ABC):
    @property
    @abstractmethod
    def measurement(self) -> pd.Series:
        """Returns a list of measurements indexed by their name."""
        pass
