from math import inf
from typing import List, Dict, Union, Tuple

import torch
from torch import Tensor

from features.param_creator_base import ParamFeaturesComposer
from models.base import StochasticModelBase, Label
from models.query_profiles import QueryProfile, AdversarialQueryProfile, NoQueriesProfile, Request
from threat_model.base import AttackResult, CriterionAttacker


class ParamFeatures:
    def __init__(self, featurizer: ParamFeaturesComposer, min_set: List[str], qps: List[AdversarialQueryProfile]):
        self.features = featurizer.make_parametric_features(qps, min_set)

        k_shape = (len(qps), len(min_set))
        self.k = torch.zeros(k_shape)

        self.min_set = min_set
        self.qps = qps

    @property
    def adv_qps(self):
        adv_qps = []
        for qp, k_i in zip(self.qps, self.k):
            for url, n in zip(self.min_set, k_i):
                for _ in range(int(n)):
                    qp.add(Request(0, url))

            adv_qps.append(qp)

        self.features.adjust_adv_qps(adv_qps)

        return adv_qps

    @property
    def x(self) -> Tensor:
        return self.features.make_features(self.k)

    @property
    def variable(self):
        return self.k

    def additional_elements(self):
        return torch.sum(self.k, dim=1)

    def update(self, d_vars: List[Tensor]):
        d_k = d_vars
        self.k.data.sub_(d_k.sign())


class GoodQueriesAttacker(CriterionAttacker):
    def __init__(
            self,
            featurizer: ParamFeaturesComposer,
            legitimate_queries: List[str],
            costs: Dict[str, float],

            max_iterations: int = 100,
    ):
        super().__init__(max_iterations)
        self.featurizer = featurizer

        self.costs = costs

        self.min_set = [legitimate_queries[0]]

    def attack(self, model: StochasticModelBase, qps: List[QueryProfile]) -> AttackResult:
        results = AttackResult()

        for qp, qp_adv in zip(qps, self.obfuscate(model, qps)):

            if isinstance(qp_adv, NoQueriesProfile):
                final_label = Label.M
                mal_prob = 1.0
                n_iters = inf
            else:
                x = self.featurizer.make_features(qp_adv, use_torch=True)
                y, mal_prob = model.predict_realize(x)
                final_label = Label.from_int(y)
                n_iters = len(qp_adv.queries) - len(qp.queries)
                if final_label == Label.B:
                    mal_prob = 1 - mal_prob

            results.add(
                n_iters, final_label, qp_adv, mal_prob=float(mal_prob)
            )

        return results

    def criterion(self, model: StochasticModelBase, x: Tensor):
        return self.costs["uncover_cost"] * model.predict_prob(x, Label.M)

    def criterion_private(self, model: StochasticModelBase, pfm: ParamFeatures) -> Tensor:
        crit = self.criterion(model, pfm.x)
        crit += self.costs["private_cost_multiplier"] * pfm.additional_elements()

        return crit

    def obfuscate(
            self,
            model: StochasticModelBase,
            qps: Union[QueryProfile, List[QueryProfile]],
            only_features=False
    ) -> Union[List[AdversarialQueryProfile], Tuple[Tensor, Tensor]]:
        if isinstance(qps, QueryProfile):
            qps = [qps]

        qps = list(map(AdversarialQueryProfile, qps))

        pfm = ParamFeatures(self.featurizer, self.min_set, qps)
        criterion = self.criterion_private(model, pfm)

        done = torch.zeros_like(pfm.variable, dtype=torch.long)

        for _ in range(self.max_attack_iters):
            d_var = -torch.ones_like(pfm.variable)
            d_var[done > 0] = 0
            pfm.update(d_var)

            new_criterion = self.criterion_private(model, pfm)
            done += (new_criterion > criterion).type(torch.long).unsqueeze(-1)
            criterion = new_criterion

        if only_features:
            attacks = criterion < self.costs["max_attack_cost"]

            return attacks, pfm.x

        else:
            adv_qps = []
            for adv_qp, crit in zip(pfm.adv_qps, criterion):
                if crit >= self.costs["max_attack_cost"]:
                    adv_qp = NoQueriesProfile(adv_qp)

                adv_qps.append(adv_qp)

            return adv_qps
