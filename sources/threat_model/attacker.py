from math import inf
from random import choice
from typing import List, Dict

from torch import Tensor
from torch.autograd import grad

from features.features_creator import FeaturesComposer
from models.base import ModelBase, Label, StochasticModelBase
from models.query_profiles import QueryProfile, Request, NoQueriesProfile, AdversarialQueryProfile
from threat_model.base import Attacker, CriterionAttacker, AttackResult


class OldGoodQueriesAttacker(Attacker):
    def __init__(self, legitimate_queries: List[str], iteration):
        super().__init__(iteration)
        self.legitimate_queries = legitimate_queries

    def obfuscate(self, model: ModelBase, query_profile: QueryProfile) -> QueryProfile:
        qp_adv = QueryProfile()

        for critical_query in query_profile.queries:
            qp_adv.add(Request(0, critical_query))

        legitimate_query = choice(self.legitimate_queries)
        qp_adv.add(Request(0, legitimate_query))

        return qp_adv


class GradientAttacker(CriterionAttacker):
    def __init__(
            self,
            featurizer: FeaturesComposer,
            legitimate_queries: List[str],
            costs: Dict[str, float],

            change_rate: float = 0.01,
            iterations: int = 20,
    ):
        super().__init__(iterations)
        self.change_rate = change_rate
        self.featurizer = featurizer
        self.costs = costs
        self.legitimate_queries = legitimate_queries

    def attack(self, model: StochasticModelBase, query_profiles: List[QueryProfile]) -> AttackResult:
        results = AttackResult()

        for qp in query_profiles:
            qp_adv, x_opt = self.obfuscate(model, qp, opt_features=True)

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
                n_iters, final_label, qp_adv, x_opt=x_opt, mal_prob=float(mal_prob)
            )

        return results

    def criterion(self, model: StochasticModelBase, x: Tensor):
        crit = self.costs["uncover_cost"] * model.predict_prob(x, Label.M)
        crit += self.private_loss(x)

        return crit

    def private_loss(self, x: Tensor) -> Tensor:
        # todo: how to define the private loss properly
        return self.costs["private_cost_multiplier"] * x.sum(1)

    def obfuscate(
            self,
            model: StochasticModelBase,
            query_profile: QueryProfile,
            opt_features=False
    ):
        query_profile = AdversarialQueryProfile(query_profile)

        x = self.featurizer.make_features(query_profile, use_torch=True)
        x.requires_grad = True

        for _ in range(self.max_attack_iters):
            d_x = grad(self.criterion(model, x), x)[0]
            x = x - self.change_rate * d_x

        qp_adv = self.project_x_to_qp(query_profile, x)
        x_final = self.featurizer.make_features(qp_adv, use_torch=True)

        if self.criterion(model, x_final) >= self.costs["max_attack_cost"]:
            qp_adv = NoQueriesProfile(qp_adv)

        if opt_features:
            return qp_adv, x
        else:
            return qp_adv

    def project_x_to_qp(self, query_profile: AdversarialQueryProfile, x_target: Tensor) -> AdversarialQueryProfile:
        # todo: generally, make this more elegant
        qp_adv = query_profile.copy()

        queries = self.legitimate_queries[:10]  # todo: choose more wisely
        query = queries.pop()

        x = self.featurizer.make_features(qp_adv, use_torch=True)
        dist = x_target.dist(x)

        while queries:
            if len(qp_adv.queries) - len(query_profile.queries) > self.costs["max_added_queries"]:
                break

            qp_adv_new = qp_adv.copy().add(Request(0, query))

            x_new = self.featurizer.make_features(qp_adv_new, use_torch=True)
            dist_new = x_target.dist(x_new)

            if dist_new < dist:
                dist = dist_new
                qp_adv = qp_adv_new
            else:
                query = queries.pop()

        return qp_adv
