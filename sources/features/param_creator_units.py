from datetime import datetime
from typing import Callable, List

import numpy as np
import torch
from pytz import timezone
from torch import Tensor

from features.param_creator_base import UnitBase, Array, ParametricUnitBase
from models.query_profiles import QueryProfile, AdversarialQueryProfile


class Histogram(UnitBase):
    def __init__(self, specificity: Callable, n_bins):
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.specificity = specificity

    def make_features(self, query_profiles: List[QueryProfile], use_torch: bool) -> Array:
        bases = torch.stack([self.basis(qp.queries, use_torch) for qp in query_profiles])
        features = self.featurize_bases(bases)
        return features

    def make_parametrized_features(self, init_qps: List[QueryProfile], urls: List[str]) -> ParametricUnitBase:
        return ParametricHistogram(self, init_qps, urls)

    def basis(self, queries, use_torch) -> Array:
        scores = list(map(lambda x: self.specificity(x), queries))
        basis, _ = np.histogram(scores, self.bins)

        if use_torch:
            basis = torch.from_numpy(basis).float()

        return basis

    def featurize_bases(self, bases: Array) -> Array:
        features = torch.zeros_like(bases)

        norm = torch.sum(bases, dim=1, keepdim=True)
        non_zero = (norm > 0).flatten()

        features[non_zero] = bases[non_zero] / norm[non_zero]
        features[~non_zero] = 0.5

        return features

    @property
    def output_len(self):
        return len(self.bins) - 1


class ParametricHistogram(ParametricUnitBase):
    def __init__(self, parent_unit: Histogram, init_qps: List[QueryProfile], urls: List[str]):
        super().__init__(init_qps, urls)
        self.parent_unit = parent_unit

        self.ingredients = torch.stack(tuple(parent_unit.basis([url], use_torch=True) for url in urls))
        self.ground = torch.stack(tuple(parent_unit.basis(qp.queries, use_torch=True) for qp in init_qps))

    def make_features(self, k: Array) -> Array:
        return self.parent_unit.featurize_bases(torch.matmul(k, self.ingredients) + self.ground)

    def project(self, k: Array):
        pass

    def adjust_adv_qps(self, adv_qps: List[QueryProfile]):
        pass

    def update(self, d_var: Tensor):
        pass


class TimeEntropy(UnitBase):
    def __init__(self, n_bins):
        self.bins = np.linspace(0, 1, n_bins + 1)

    def make_features(self, query_profiles: List[QueryProfile], use_torch: bool) -> Array:
        bases = torch.stack(tuple([self.basis(qp, use_torch) for qp in query_profiles]))
        features = self.featurize_bases(bases)
        return features

    def make_parametrized_features(self, init_qps: List[QueryProfile], urls: List[str]) -> ParametricUnitBase:
        return ParametricTimeEntropy(self, init_qps, urls)

    def basis(self, qp: QueryProfile, use_torch) -> Array:
        def relative_day_time(time: int):
            dt = datetime.fromtimestamp(time, timezone("CET"))
            return (dt.hour + 1) / 24

        times = map(lambda x: x.time, qp.requests)
        rel_times = list(map(relative_day_time, times))
        basis, _ = np.histogram(rel_times, self.bins)

        if use_torch:
            basis = torch.from_numpy(basis).float()

        return basis

    def featurize_bases(self, bases: Array) -> Array:
        features = torch.zeros_like(bases)

        norm = torch.sum(bases, dim=1, keepdim=True)
        non_zero = (norm > 0).flatten()

        features[non_zero] = bases[non_zero] / norm[non_zero]
        features[~non_zero] = 0.5

        return features

    @property
    def output_len(self):
        return len(self.bins) - 1


class ParametricTimeEntropy(ParametricUnitBase):
    def __init__(self, parent_unit: TimeEntropy, init_qps: List[QueryProfile], urls: List[str]):
        super().__init__(init_qps, urls)
        self.parent_unit = parent_unit
        self.n_bins = parent_unit.output_len

        self.var = torch.ones(len(init_qps), self.n_bins, requires_grad=True) / self.n_bins

    def make_features(self, k: Array) -> Array:
        return self.var

    def project(self, k: Array):
        self.var.data[self.var.data < 0] = 0
        self.var.data = self.parent_unit.featurize_bases(self.var.data)

        count_exp = torch.sum(k, dim=1, keepdim=True) \
                    + torch.tensor(
            [[len(qp.requests)] for qp in self.init_qps]
            , dtype=torch.float)
        n_est = self.estimate_n(count_exp)
        self.var.data = self.parent_unit.featurize_bases(n_est)

    def estimate_n(self, count_exp: Tensor):
        n_approx = count_exp * self.var
        n_est = n_approx.floor()

        count_delta = count_exp - torch.sum(n_est, dim=1, keepdim=True)
        # n_delta = n_approx - n_est
        n_est[:, 0] += count_delta.flatten()

        # while non_opt_idx.all():
        #     idx = torch.argmax(n_delta[non_opt_idx], dim=1)
        #     n_est[non_opt_idx][[range(len(idx)), idx]] += 1
        #
        #     count_delta[non_opt_idx] -= 1
        #     n_delta = n_approx - n_est
        #     non_opt_idx = (count_delta != 0)

        return n_est

    def adjust_adv_qps(self, adv_qps: List[AdversarialQueryProfile]):
        count_exp = torch.tensor([[len(qp.requests)] for qp in adv_qps], dtype=torch.float)
        # print(count_exp.shape)
        # print()
        estimated_entropy_n = self.estimate_n(count_exp)
        for adv_qp, n_est in zip(adv_qps, estimated_entropy_n):
            requests = iter(adv_qp.requests)
            # print(len(adv_qp.requests))
            # print(n_est)
            # print(torch.sum(n_est))

            for bin, n in enumerate(list(n_est)):
                # print(bin, n)
                for _ in range(int(n)):
                    request = next(requests)
                    request.time = 24 * bin / self.n_bins

    def update(self, d_var: Tensor):
        self.var.data.sub_(d_var)


class Count(UnitBase):
    def make_features(self, query_profiles: List[QueryProfile], use_torch: bool) -> Array:
        bases = torch.tensor([[len(qp.queries)] for qp in query_profiles], dtype=torch.float)
        return torch.log(bases)

    def make_parametrized_features(self, init_qps: List[QueryProfile], urls: List[str]) -> ParametricUnitBase:
        return ParametricCount(init_qps, urls)

    @property
    def output_len(self):
        return 1


class ParametricCount(ParametricUnitBase):
    def __init__(self, init_qps: List[QueryProfile], urls: List[str]):
        super().__init__(init_qps, urls)
        self.ground = torch.tensor([[len(qp.queries)] for qp in init_qps], dtype=torch.float)

    def make_features(self, k: Array) -> Array:
        return torch.log(self.ground + torch.sum(k, dim=1, keepdim=True))

    def project(self, k: Array):
        pass

    def adjust_adv_qps(self, adv_qps: List[QueryProfile]):
        pass

    def update(self, d_var: Tensor):
        pass
