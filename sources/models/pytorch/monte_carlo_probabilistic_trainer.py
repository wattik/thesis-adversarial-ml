from typing import List

import torch
from torch.autograd import grad

from containers import RotDeque
from features.features_creator import FeaturesComposer
from models.base import Label, StochasticModelBase
from models.pytorch.autograd import jacobian
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

    ):
        self.phi_groups = list(phi_groups)
        self.attacker = attacker
        self.featurizer = featurizer
        self.model = model

        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def batch_step(
            self,
            benign_query_profiles: RotDeque,
            malicious_query_profiles: RotDeque,
            prob_benign: float
    ):
        updates_summed = [torch.zeros_like(phi) for phi in self.phi_groups]

        for batch_i in range(self.batch_size):
            label = Label.draw(prob_benign)

            if label == Label.B:
                qp = benign_query_profiles.next()
            else:
                qp = malicious_query_profiles.next()
                # qp = self.attacker.obfuscate(self.model, qp)

            x, y = self.featurizer.make_features(qp, label, use_torch=True)
            x.requires_grad = True

            y_alt_prob = torch.log(self.model.predict_prob(x, label))
            delta_phi_groups = tuple(self.get_delta_phi(x, y_alt_prob, y))

            for update, delta_phi in zip(updates_summed, delta_phi_groups):
                update += delta_phi

        for phi, delta_phi in zip(self.phi_groups, updates_summed):
            phi.data.add_(self.learning_rate / self.batch_size * delta_phi)

    def get_delta_phi(self, x: torch.Tensor, y_alt_prob: torch.Tensor, y: torch.Tensor):
        if len(y) > 1:
            raise ValueError("This version only supports single sample delta phi computation."
                             " Received samples with more sample.")

        # Loss is non-zero and the update depends on the detector gradient
        elif y == Label.B.to_int():
            return self.get_delta_phi_simple(y_alt_prob)
        elif y == Label.M.to_int():
            # return self.get_delta_phi_simple(y_alt_prob)
            return self.get_delta_phi_adversarial(x, y_alt_prob)

        else:
            raise Exception("WHAT!")

    def get_delta_phi_simple(self, y_alt_prob: torch.Tensor):
        return [g for g in grad(y_alt_prob, self.phi_groups)]

    def get_delta_phi_adversarial(self, x: torch.Tensor, y_alt_prob: torch.Tensor):
        for phi in self.phi_groups:
            d_phi = grad(y_alt_prob, phi, retain_graph=True)[0]
            d_x = grad(y_alt_prob, x, retain_graph=True)[0]

            r = self.attacker.criterion(self.model, x)
            f = grad(r, x, create_graph=True)[0]

            jacob = - jacobian(f, phi) / 2
            # inv_f = torch.inverse(jacobian(f, x).squeeze())
            # jacob = - jacobian(f, phi).matmul(inv_f)

            adv_jacob = jacob.matmul(d_x.flatten()).squeeze()

            yield d_phi + adv_jacob
