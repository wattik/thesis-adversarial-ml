from typing import List

import torch
from torch.autograd import grad

from containers import RotDeque
from features.creator import FeaturesComposer
from models.base import Label, StochasticModelBase
from models.query_profiles import NoQueriesProfile
from threat_model.base import CriterionAttacker


class MonteCarloTrainer:
    def __init__(
            self,
            phi_groups: List[torch.Tensor],
            attacker: CriterionAttacker,
            featurizer: FeaturesComposer,
            model: StochasticModelBase,

            learning_rate=0.1,
            batch_size=32,
            fp_thresh=0.01,
    ):
        self.phi_groups = list(phi_groups)
        self.attacker = attacker
        self.featurizer = featurizer
        self.model = model

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.fp_thresh = fp_thresh

    def batch_step(
            self,
            benign_query_profiles: RotDeque,
            malicious_query_profiles: RotDeque,
    ):
        lam = float(self.model.model.lam)

        _, updates_mal = self.get_class_update(malicious_query_profiles, Label.M)
        error_ben, updates_ben = self.get_class_update(benign_query_profiles, Label.B)

        self.model.model.lam.data.add_(self.learning_rate * (error_ben - self.fp_thresh))

        for phi, delta_phi_mal, delta_phi_ben in zip(self.phi_groups, updates_mal, updates_ben):
            phi.data.sub_(self.learning_rate * (delta_phi_mal + lam * delta_phi_ben))

    def get_class_update(self, query_profiles: RotDeque, label: Label):
        updates = [torch.zeros_like(phi) for phi in self.phi_groups]
        error = 0.0

        for _ in range(self.batch_size):
            qp = query_profiles.next()

            if label == Label.M:
                qp = self.attacker.obfuscate(self.model, qp)

                if isinstance(qp, NoQueriesProfile):
                    continue  # No cost.

            x, y = self.featurizer.make_features(qp, label, use_torch=True)
            # x.requires_grad = True

            # todo: what if changed to: gradient ascent and logitized
            y_alt_prob = 1 - self.model.predict_prob(x, label)
            # ERROR
            error += float(torch.sum(y_alt_prob)) / self.batch_size
            # UPDATE
            delta_phi_groups = tuple(self.get_delta_phi(x, y_alt_prob, y))
            for update, delta_phi in zip(updates, delta_phi_groups):
                update += delta_phi / self.batch_size

        return error, updates

    def get_delta_phi(self, x: torch.Tensor, y_alt_prob: torch.Tensor, y: torch.Tensor):
        if len(y) > 1:
            raise ValueError("This version only supports single sample delta phi computation."
                             "Received tensors with more sample.")

        # Loss is non-zero and the update depends on the detector gradient
        elif y == Label.B.to_int():
            return self.get_delta_phi_simple(y_alt_prob)
        elif y == Label.M.to_int():
            return self.get_delta_phi_simple(y_alt_prob)
            # return self.get_delta_phi_adversarial(x, y_alt_prob)

        else:
            raise Exception("WHAT!")

    def get_delta_phi_simple(self, y_alt_prob: torch.Tensor):
        return [g for g in grad(y_alt_prob, self.phi_groups)]

    # def get_delta_phi_adversarial(self, x: torch.Tensor, y_alt_prob: torch.Tensor):
    #     for phi in self.phi_groups:
    #         d_phi = grad(y_alt_prob, phi, retain_graph=True)[0]
    #         d_x = grad(y_alt_prob, x, retain_graph=True)[0]
    #
    #         r = self.attacker.criterion(self.model, x)
    #         f = grad(r, x, create_graph=True)[0]
    #
    #         inv_f = torch.inverse(jacobian(f, x).squeeze())
    #         jacob = - jacobian(f, phi).matmul(inv_f)
    #
    #         adv_jacob = jacob.matmul(d_x.flatten()).squeeze()
    #
    #         yield d_phi + adv_jacob
