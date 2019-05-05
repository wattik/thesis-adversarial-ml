from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as func

from features.features_creator import FeaturesComposer
from models.base import Label, StochasticModelBase
from models.pytorch.langrange_trainer import MonteCarloTrainer
from models.query_profiles import QueryProfile
from models.sampling.sampler import TensorSampler, Sampler
from threat_model.base import CriterionAttacker

torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self, n_features, lambda_init: float):
        super(Net, self).__init__()

        self.dense1 = nn.Linear(n_features, n_features * 10)
        self.dense2 = nn.Linear(n_features * 10, n_features * 10)
        self.dense3 = nn.Linear(n_features * 10, n_features * 5)
        self.dense4 = nn.Linear(n_features * 5, n_features * 2)
        #
        self.dense_final = nn.Linear(n_features * 2, 1)

        # self.lam = torch.tensor(lambda_init, requires_grad=True, dtype=torch.float)
        self.lam = lambda_init

    def forward(self, x):
        x = func.selu(self.dense1(x))
        x = func.selu(self.dense2(x))
        x = func.selu(self.dense3(x))
        x = func.selu(self.dense4(x))

        x = self.dense_final(x)
        # x = func.dropout(x, training=self.training)
        x = 1 / (1 + torch.exp(-x))

        return x


class MonteCarloNet(StochasticModelBase):
    def __init__(
            self,
            featurizer: FeaturesComposer,
            attacker: CriterionAttacker,

            batch_loops=100,
            lambda_init=9.0,
            batch_size=32,
            fp_threshold=0.01,
            lr=0.1,
            lambda_lr=10.0
    ):
        self.batch_loops = batch_loops

        self.attacker = attacker
        self.featurizer = featurizer
        self.model = Net(self.featurizer.n_features, lambda_init)

        self.trainer = MonteCarloTrainer(
            self.model.parameters(),
            self.attacker,
            self.featurizer,
            self,
            batch_size=batch_size,
            fp_thresh=fp_threshold,
            learning_rate=lr,
            lambda_learning_rate=lambda_lr
        )

    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        benign_samples = TensorSampler([
            self.featurizer.make_features(qp, use_torch=True) for qp, l in zip(query_profiles, labels) if l == Label.B
        ])
        malicious_query_profiles = Sampler([
            qp for qp, l in zip(query_profiles, labels) if l == Label.M
        ])

        self.fit_samplers(benign_samples, malicious_query_profiles)

    def fit_samplers(self, benign_samples: TensorSampler, malicious_query_profiles: Sampler):
        for i in range(self.batch_loops):
            if i % int(self.batch_loops / 10) == 0:
                print("*", end="", flush=True)

            self.trainer.batch_step(
                benign_samples,
                malicious_query_profiles,
            )

        print()

    def predict(self, query_profiles: List[QueryProfile]):
        X = self.featurizer.make_features(query_profiles, use_torch=True)
        decisions, _ = self.predict_realize(X)
        return [Label.from_int(d) for d in decisions]

    def predict_realize(self, x: torch.Tensor):
        mal_prob = self.model(x)
        decisions = (torch.rand_like(mal_prob) < mal_prob).long()
        mal_prob[decisions == 0] = 1 - mal_prob[decisions == 0]
        return decisions, mal_prob

    def predict_prob(self, x: torch.Tensor, label: Label):
        if label == Label.M:
            return self.model(x).flatten()
        else:
            return 1 - self.model(x).flatten()

    def save(self, path: str):
        torch.save(self.model, path)

    def load(self, path: str):
        self.model = torch.load(path)
