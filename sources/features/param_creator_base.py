from abc import ABC, abstractmethod
from typing import NewType, Union, List

import torch
from numpy import ndarray
from torch import Tensor

from models.base import Label
from models.query_profiles import QueryProfile
from normalization import Normalizer

Array = NewType("Array", Union[Tensor, ndarray])


class ParametricUnitBase(ABC):
    def __init__(self, init_qps: List[QueryProfile], urls: List[str]):
        self.urls = urls
        self.init_qps = init_qps

    @abstractmethod
    def make_features(self, k: Array) -> Array:
        pass

    @abstractmethod
    def project(self, k: Array):
        pass

    @abstractmethod
    def adjust_adv_qps(self, adv_qps: List[QueryProfile]):
        pass

    @abstractmethod
    def update(self, d_var: Tensor):
        pass


class UnitBase(ABC):
    """
    If needed, bases itself can be a variable with grad and self.project(estimate_n) will project it back.
    Not sure if this is complete though.
    """

    @abstractmethod
    def make_features(self, query_profiles: List[QueryProfile], use_torch: bool) -> Array:
        pass

    @abstractmethod
    def make_parametrized_features(self, init_qps: List[QueryProfile], urls: List[str]) -> ParametricUnitBase:
        pass

    # @abstractmethod
    # def construct_bases(self, init_qps: List[QueryProfile], urls: List[str], use_torch: bool) -> Tuple[Array, Array]:
    #     pass

    # @abstractmethod
    # def construct_param_features(self, ground: Array, k: Array, ingredients: Array) -> Array:
    #     pass

    @property
    @abstractmethod
    def output_len(self):
        pass


class ParamFeaturesComposer:
    def __init__(self, features_creators: List[UnitBase]):
        self.units = features_creators
        self.normalizer = Normalizer(self.n_features)

    def fit_normalizer(self, query_profiles: List[QueryProfile]):
        x = self.make_features(query_profiles, use_torch=True)
        self.normalizer.fit(x)

    @property
    def n_features(self):
        return sum([f.output_len for f in self.units])

    def make_features(
            self,
            query_profiles: Union[List[QueryProfile], QueryProfile],
            labels: Union[List[Label], Label] = None,
            use_torch=False,
    ):
        if isinstance(query_profiles, QueryProfile):
            query_profiles = [query_profiles]

        if isinstance(labels, Label):
            labels = [labels]

        # Actual feature composing
        X = torch.cat(tuple(f.make_features(query_profiles, use_torch=True) for f in self.units), dim=1)
        X = self.normalizer.transform(X)
        # Features Composed

        if not use_torch:
            X = X.numpy()

        res = X

        if labels:
            y = torch.tensor([label.to_int() for label in labels])
            if not use_torch:
                y = y.numpy()

            res = (X, y)

        return res

    def make_parametric_features(self, init_qps: List[QueryProfile], urls: List[str]) -> "ParametricFeatureMap":
        return ParametricFeatureMap([
            unit.make_parametrized_features(init_qps, urls) for unit in self.units
        ], self.normalizer)


class ParametricFeatureMap:
    def __init__(self, units: List[ParametricUnitBase], normalizer: Normalizer):
        self.normalizer = normalizer
        self.units = units

    def make_features(self, k: Array) -> Array:
        return self.normalizer.transform(
            torch.cat([unit.make_features(k) for unit in self.units], dim=1)
        )

    def project(self, k: Array):
        for unit in self.units:
            unit.project(k)

    def update(self, d_vars: List[Tensor]):
        d_vars = iter(d_vars)
        for unit in self.units:
            if hasattr(unit, "var"):
                unit.update(next(d_vars))

    @property
    def variables(self) -> List[Tensor]:
        variables = []
        for unit in self.units:
            if hasattr(unit, "var"):
                variables.append(unit.var)

        return variables

    def adjust_adv_qps(self, adv_qps: List[QueryProfile]):
        for unit in self.units:
            unit.adjust_adv_qps(adv_qps)
