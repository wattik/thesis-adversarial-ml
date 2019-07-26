from typing import List

import torch
from torch.autograd import grad

from features.features_creator import FeaturesComposer
from models.base import Label, StochasticModelBase
from models.pytorch.autograd import jacobian
from models.sampling.sampler import Sampler
from threat_model.histogram_attacker import FGSMAttacker


class MonteCarloTrainer:
    def __init__(
            self,
            phi_groups: List[torch.Tensor],
            attacker: FGSMAttacker,
            featurizer: FeaturesComposer,
            model: StochasticModelBase,

            learning_rate=0.1,
            batch_size=32,
            fp_thresh=0.01,
            lambda_learning_rate=10.0,
    ):
        self.phi_groups = list(phi_groups)
        self.attacker = attacker
        self.featurizer = featurizer
        self.model = model

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.fp_thresh = fp_thresh
        self.lambda_learning_rate = lambda_learning_rate

    def batch_step(
            self,
            benign_samples: Sampler,
            malicious_samples: Sampler,
    ):
        lam = float(self.model.model.lam)

        error_ben, updates_ben = self.get_class_update_simple(benign_samples, Label.B)
        error_mal, updates_mal = self.get_class_update_simple(malicious_samples, Label.M)

        # LAMBDA update
        self.model.model.lam = min([
            max([
                0.001,
                lam + self.lambda_learning_rate * (float(error_ben) - self.fp_thresh)
            ]),
            2000.0
        ])

        # WEIGHTS update
        for phi, delta_phi_mal, delta_phi_ben in zip(self.phi_groups, updates_mal, updates_ben):
            phi.data.add_(self.learning_rate * (delta_phi_mal + lam * delta_phi_ben))

    def get_class_update_simple(self, samples: Sampler, label: Label):
        if label == Label.B:
            x = samples.next(self.batch_size)
        else:  # Label.M
            qps = samples.next(self.batch_size)
            attacks, x = self.attacker.obfuscate(self.model, qps, only_features=True)

            if (attacks == 0).all():
                # No one attacked.
                return 0, [torch.zeros_like(phi) for phi in self.phi_groups]

            x = x[attacks.flatten(), :]

        n = float(len(x))

        y_prob = self.model.predict_prob(x, label)
        error = torch.sum(1 - y_prob) / n
        criterion = torch.sum(torch.log(y_prob))

        updates = [
            delta_phi / n
            for delta_phi in self.get_delta_phi_simple(x, criterion)
        ]

        updates = map(lambda d: torch.where(torch.isnan(d), torch.zeros_like(d), d), updates)
        return error, list(updates)

    def get_delta_phi_simple(self, x: torch.Tensor, y_alt_prob: torch.Tensor):
        return [g for g in grad(y_alt_prob, self.phi_groups)]

    #     def get_class_update_adversarial(self, samples: RotDeque, label: Label):
    #         updates = [torch.zeros_like(phi) for phi in self.phi_groups]
    #         error = 0.0
    #
    #         for _ in range(self.batch_size):
    #
    #             if label == Label.M:
    #                 qp = samples.next()[0]
    #                 qp = self.attacker.obfuscate(self.model, qp)
    #
    #                 if isinstance(qp, NoQueriesProfile):
    #                     continue  # No cost. The Attacker does not attack.
    #
    #                 x = self.featurizer.make_features(qp, use_torch=True)
    #                 get_delta_phi = self.get_delta_phi_simple
    #
    #             else:  # Label.B:
    #                 x = samples.next()[0]
    #                 get_delta_phi = self.get_delta_phi_simple
    #
    #             x.requires_grad = True
    #
    #             # todo: what if changed to: gradient ascent and logitized
    #             y_alt_prob = 1 - self.model.predict_prob(x, label)
    #             # ERROR
    #             error += float(torch.sum(y_alt_prob)) / self.batch_size
    #             # UPDATE
    #             delta_phi_groups = tuple(get_delta_phi(x, y_alt_prob))
    #             for update, delta_phi in zip(updates, delta_phi_groups):
    #                 update += delta_phi / self.batch_size
    #
    #         return error, updates
    #
    def get_delta_phi_adversarial(self, x: torch.Tensor, y_alt_prob: torch.Tensor):
        d_x = grad(y_alt_prob, x, retain_graph=True)[0]

        f = grad(self.attacker.criterion(self.model, x), x, create_graph=True)[0]

        jacob_x = jacobian(f, x).squeeze()
        inv_jacob_x = inv(jacob_x)

        if torch.sum(inv(inv_jacob_x) - jacob_x) > 1e-1 * jacob_x.numel():
            # Unstable inverse
            for phi in self.phi_groups:
                d_phi = grad(y_alt_prob, phi, retain_graph=True)[0]
                yield d_phi

        else:
            # Stable Inverse
            for phi in self.phi_groups:
                d_phi = grad(y_alt_prob, phi, retain_graph=True)[0]
                jacob_phi = jacobian(f, phi)
                adv_jacob = torch.matmul(jacob_phi, inv_jacob_x)

                yield d_phi - torch.matmul(adv_jacob, d_x.flatten()).squeeze()


def inv(tensor: torch.Tensor):
    return torch.from_numpy(np.linalg.inv(tensor.numpy()))
