from typing import List

import torch
from torch.autograd import grad

from containers import RotDeque
from features.creator import FeaturesComposer
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

    ):
        self.phi_groups = list(phi_groups)
        self.attacker = attacker
        self.featurizer = featurizer
        self.model = model

        self.learning_rate = 1
        self.batch_size = 32

    def batch_step(
            self,
            benign_query_profiles: RotDeque,
            malicious_query_profiles: RotDeque,
            prob_benign: float
    ):
        for batch_i in range(self.batch_size):
            label = Label.draw(prob_benign)

            if label == Label.B:
                qp = benign_query_profiles.next()
            else:
                qp = malicious_query_profiles.next()
                qp = self.attacker.obfuscate(self.model, qp)

            x, y = self.featurizer.make_features(qp, label, use_torch=True)
            x.requires_grad = True

            y_pred, y_prob = self.model.predict_realize(x)
            delta_phi_groups = tuple(self.get_delta_phi(x, y_pred, y_prob, y))

            for phi, delta_phi in zip(self.phi_groups, delta_phi_groups):
                phi.data.sub_(self.learning_rate / self.batch_size * delta_phi)

    def get_delta_phi(self, x: torch.Tensor, y_pred: torch.Tensor, y_prob: torch.Tensor, y: torch.Tensor):
        if len(y) > 1:
            raise ValueError("This version only supports single sample delta phi computation."
                             " Received tensors with more sample.")

        # Loss is zero
        if y_pred == y:
            return [torch.zeros_like(phi) for phi in self.phi_groups]

        # Loss is non-zero and the update depends on the detector gradient
        elif y == Label.B.to_int():
            return self.get_delta_phi_ben(y_prob)
        elif y == Label.M.to_int():
            return self.get_delta_phi_ben(y_prob)
            # return self.get_delta_phi_adversarial(x, y_prob)
        else:
            raise Exception("WHAT!")

    def get_delta_phi_ben(self, y_prob: torch.Tensor):
        return ((g / y_prob).reshape_as(g) for g in grad(y_prob, self.phi_groups))

    def get_delta_phi_mal(self, x: torch.Tensor, y_prob: torch.Tensor):
        for phi in self.phi_groups:
            d_phi = grad(y_prob, phi, retain_graph=True)[0]
            d_x = grad(y_prob, x, retain_graph=True)[0]

            r = self.attacker.criterion(self.model, x)
            f = grad(r, x, create_graph=True)[0]

            div = jacobian(f, x).squeeze().diag()

            if (div == 0).any():
                print("DIV by ZERO, exploding gradient.")
                exit()

            jacob = jacobian(f, phi) / div

            adv_jacob = jacob.matmul(d_x.flatten())

            yield ((d_phi - adv_jacob) / y_prob).reshape_as(d_phi)
