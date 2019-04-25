from itertools import tee
from typing import List, Dict, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.autograd import grad

from features.creator import FeaturesComposer
from models.base import StochasticModelBase, Label
from models.query_profiles import QueryProfile, AdversarialQueryProfile, NoQueriesProfile, Request
from probability import Probability
from threat_model.attacker import GradientAttacker


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def make_min_set(specificity: Probability, legitimate_queries, bins):
    legitimate_queries += ["UKNOWN-URL.CZ"]
    n = len(bins) - 1  # n_bins
    basis = [None] * n

    for i, (lb, up) in enumerate(pairwise(bins)):
        for q in legitimate_queries:
            f = specificity(q)
            if lb <= f < up:
                basis[i] = q
                break

    return basis


class HistogramAttacker(GradientAttacker):
    def __init__(
            self,
            featurizer: FeaturesComposer,
            legitimate_queries: List[str],
            costs: Dict[str, float],
            specificity: Probability,

            change_rate: float = 0.01,
            max_iterations: int = 20,
    ):
        super().__init__(featurizer, legitimate_queries, costs, change_rate, max_iterations)
        self.specificity = specificity

        self.n_bins = featurizer.n_features
        self.min_set = make_min_set(specificity, legitimate_queries, np.linspace(0, 1, self.n_bins + 1))

    def crit(self, model: StochasticModelBase, query_profile: QueryProfile, count_vector: Tensor):
        x, n = self.construct_x_n(query_profile, count_vector)
        crit = self.costs["uncover_cost"] * model.predict_prob(x, Label.M)
        crit += self.costs["private_cost_multiplier"] * torch.sum(n, dim=1)

        return crit

    def obfuscate(
            self,
            model: StochasticModelBase,
            query_profile: QueryProfile,
            opt_features=False
    ) -> Union[
        AdversarialQueryProfile,
        Tuple[AdversarialQueryProfile, Tensor]
    ]:
        query_profile = AdversarialQueryProfile(query_profile)

        n = torch.zeros(self.n_bins, requires_grad=True)
        criterion = self.crit(model, query_profile, n)

        for _ in range(self.max_attack_iters):
            d_n = grad(criterion, n)[0]
            n = n - self.change_rate * d_n

            # project n to allowed numbers (n>0, n \in N)
            n[n < 0] = 0
            n.round_()

            criterion = self.crit(model, query_profile, n)

        x, _ = self.construct_x_n(query_profile, n)
        qp_adv = self.construct_qp_from_n(query_profile, n)

        if criterion >= self.costs["max_attack_cost"]:
            qp_adv = NoQueriesProfile(qp_adv)

        if opt_features:
            return qp_adv, x
        else:
            return qp_adv

    def construct_x_n(self, query_profile: QueryProfile, count_vector: Tensor):
        n = self.featurizer.make_features(query_profile, use_torch=True) * len(query_profile.queries)
        n += count_vector
        x = n / torch.sum(n, dim=1)
        return x, n

    def construct_qp_from_n(self, query_profile: QueryProfile, count_vector: Tensor):
        for url, n in zip(self.min_set, count_vector):
            for _ in range(int(n)):
                query_profile.add(Request(0, url))

        return query_profile
