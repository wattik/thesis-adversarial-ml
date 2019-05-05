from abc import ABC, abstractmethod
from enum import Enum
from random import random
from typing import List

from models.query_profiles import QueryProfile


class Label(Enum):
    B = "<Benign>"
    M = "<Malicious>"

    def to_int(self):
        if self == Label.M:
            return 1
        else:
            return 0

    @classmethod
    def from_int(cls, n: int):
        if n == 1:
            return Label.M
        else:
            return Label.B

    @classmethod
    def draw(cls, prob_benign: float):
        if random() < prob_benign:
            return Label.B
        else:
            return Label.M


class ModelBase(ABC):
    @abstractmethod
    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        pass

    @abstractmethod
    def predict(self, query_profiles: List[QueryProfile]) -> List[Label]:
        pass

    def predict_one(self, query_profile: QueryProfile) -> Label:
        return self.predict([query_profile])[0]

    @abstractmethod
    def save(self, path: str):
        pass


class StochasticModelBase(ModelBase):
    @abstractmethod
    def predict_realize(self, x):
        pass

    @abstractmethod
    def predict_prob(self, x, label: Label):
        pass
