from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

import numpy as np

from models.base import ModelBase, Label
from models.query_profiles import QueryProfile

AttackInstance = namedtuple("AttackInstance", [
    "n_iters",
    "final_label",
    "qp_adv",
    "mal_prob",
    "x_opt",
])


class AttackResult:
    def __init__(self):
        self.results = []

    def add(self, n_iters, final_label, qp_adv, mal_prob=None, x_opt=None):
        mal_prob = mal_prob if mal_prob else final_label.to_int()
        res_instance = AttackInstance(n_iters, final_label, qp_adv, mal_prob, x_opt)
        self.results.append(res_instance)

    def get_query_profiles(self):
        return [res.qp_adv for res in self.results]

    def get_labels(self):
        return [res.final_label for res in self.results]

    def get_undetected_query_profiles(self):
        return [res.qp_adv for res in self.results if res.final_label == Label.B]

    def get_optimal_feature_vectors(self):
        return [res.x_opt for res in self.results]

    def get_malicious_probabilities(self):
        return [res.mal_prob for res in self.results]

    @property
    def total_attacks_number(self):
        return len(self.results)

    @property
    def no_attack_success_rate(self):
        n = [res for res in self.results if res.n_iters == 0 and res.final_label == Label.B]
        return len(n) / len(self.results)

    @property
    def attack_success_rate(self):
        n = [res for res in self.results if res.final_label == Label.B]
        return len(n) / len(self.results)

    @property
    def mean_attack_step(self):
        n = [res.n_iters for res in self.results if res.final_label == Label.B]

        if not n:
            return np.nan

        return np.mean(n)


class Attacker(ABC):
    def __init__(self, max_attack_iterations: int = 100):
        self.max_attack_iters = max_attack_iterations

    def attack(self, model: ModelBase, query_profiles: List[QueryProfile]) -> AttackResult:
        results = AttackResult()

        for label, qp, n_iter in zip(*self._batch_attack(query_profiles, model)):
            results.add(n_iter, label, qp)

        return results

    def _single_attack(self, model: ModelBase, qp_init: QueryProfile):
        label: Label
        qp = qp_init

        for i in range(0, self.max_attack_iters + 1):
            label = model.predict([qp])[0]

            if label == Label.B:
                break

            qp = self.obfuscate(model, qp)

        return label, qp, i

    def _batch_attack(self, qps_init: List[QueryProfile], model: ModelBase):
        n = len(qps_init)

        final_labels: List[Label] = [None] * n
        final_qps: List[QueryProfile] = [None] * n
        iters_bag: List[int] = [None] * n

        qps = qps_init[:]
        idx = list(range(n))

        for iter in range(0, self.max_attack_iters + 1):
            if not qps:
                break

            labels = model.predict(qps)

            succ_idx = [i for i, l in enumerate(labels) if l == Label.B]
            non_succ_idx = [i for i, l in enumerate(labels) if l == Label.M]

            for i in succ_idx:
                final_labels[idx[i]] = Label.B
                final_qps[idx[i]] = qps[i]
                iters_bag[idx[i]] = iter

            idx = [idx[i] for i in non_succ_idx]
            qps = [self.obfuscate(model, qps[i]) for i in non_succ_idx]

        for i, qp in zip(idx, qps):
            final_labels[i] = Label.M
            final_qps[i] = qp
            iters_bag[i] = self.max_attack_iters

        return final_labels, final_qps, iters_bag

    @abstractmethod
    def obfuscate(self, model: ModelBase, query_profile: QueryProfile) -> QueryProfile:
        pass


class CriterionAttacker(Attacker, ABC):
    @abstractmethod
    def criterion(self, model, x):
        pass
