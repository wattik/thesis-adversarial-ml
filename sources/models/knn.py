from typing import List

import torch
from torch.autograd import grad

from features.features_creator import FeaturesComposer
from models.base import Label, StochasticModelBase
from models.query_profiles import QueryProfile
from models.sampling.sampler import TensorSampler, Sampler


class KNN(StochasticModelBase):
    def __init__(
            self,
            featurizer: FeaturesComposer,
            k: int = 5,
            fp_thresh=0.01,
            lr=0.1,
            validation_ratio=0.2,
            alpha_init=1.0,
            max_iter=100,
    ):
        self.ftol = fp_thresh / 10
        self.featurizer = featurizer

        self.ground_set = None
        self.validation_set = None
        self.alpha = torch.tensor(alpha_init, requires_grad=True, dtype=torch.float)

        self.max_iter = max_iter
        self.k = k
        self.fp_thresh = fp_thresh
        self.lr = lr
        self.validation_ratio = validation_ratio

    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        benign_samples = TensorSampler([
            self.featurizer.make_features(qp, use_torch=True) for qp, l in zip(query_profiles, labels) if l == Label.B
        ])
        malicious_query_profiles = Sampler([
            qp for qp, l in zip(query_profiles, labels) if l == Label.M
        ])

        self.fit_samplers(benign_samples, malicious_query_profiles)

    def fit_samplers(self, benign_samples: TensorSampler, malicious_query_profiles: Sampler):
        total_samples = len(benign_samples)
        validation_size = int(self.validation_ratio * total_samples)
        training_size = total_samples - validation_size

        self.ground_set = benign_samples.next(training_size)
        self.validation_set = benign_samples.next(validation_size)

        for i in range(self.max_iter):
            print(i)
            error = (self.fp_thresh - torch.mean(self.predict_prob(self.validation_set, Label.M))) ** 2
            print(error * 100)
            print(self.alpha)
            print()

            if error < (self.ftol**2):
                break

            d_alpha = grad(error, self.alpha)[0]
            self.alpha.data.sub_(self.lr * d_alpha.sign())

    def mean_distances(self, x):
        mean_distances = []
        for x_i in x:
            d = torch.norm(self.ground_set - x_i.unsqueeze(0), dim=1)
            d, _ = torch.topk(d, self.k, largest=False)
            mean_distances.append(torch.mean(d))

        return torch.stack(mean_distances)

    def predict_prob(self, x, label: Label):
        dist = self.mean_distances(x)
        prob_ben = torch.exp(- ((dist / self.alpha) ** 2))

        if label == Label.B:
            return prob_ben
        else:
            return 1 - prob_ben

    def predict_realize(self, x):
        mal_prob = self.predict_prob(x, Label.M)
        decisions = (torch.rand_like(mal_prob) < mal_prob).long()
        mal_prob[decisions == 0] = 1 - mal_prob[decisions == 0]
        return decisions, mal_prob

    def predict(self, query_profiles: List[QueryProfile]) -> List[Label]:
        X = self.featurizer.make_features(query_profiles, use_torch=True)
        decisions, _ = self.predict_realize(X)
        return [Label.from_int(d) for d in decisions]
