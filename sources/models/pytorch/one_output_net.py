from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as func

from containers import RotDeque
from features.creator import FeaturesComposer
from models.base import Label, StochasticModelBase
from models.pytorch.monte_carlo_probabilistic_trainer import MonteCarloTrainer
from models.query_profiles import QueryProfile
from threat_model.base import CriterionAttacker

torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)


class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()

        # self.dense1 = nn.Linear(n_features, n_features * 2)
        # self.dense2 = nn.Linear(n_features * 2, n_features * 2)
        # self.dense3 = nn.Linear(n_features * 2, n_features * 2)
        # self.dense4 = nn.Linear(n_features * 2, n_features * 2)

        self.dense_final = nn.Linear(n_features, 1)

    def forward(self, x):
        # x = func.selu(self.dense1(x))
        # x = func.leaky_relu(self.dense2(x))
        # x = func.selu(self.dense3(x))
        # x = func.selu(self.dense4(x))

        # x = func.selu(self.dense_final(x))
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
            lambdas=None,
    ):
        self.lambdas = lambdas if lambdas else [9, 99, 999]
        self.batch_loops = batch_loops

        self.attacker = attacker
        self.featurizer = featurizer
        self.model = Net(self.featurizer.n_features)

        self.trainer = MonteCarloTrainer(self.model.parameters(), self.attacker, self.featurizer, self)

    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        benign_query_profiles = RotDeque(qp for qp, l in zip(query_profiles, labels) if l == Label.B)
        malicious_query_profiles = RotDeque(qp for qp, l in zip(query_profiles, labels) if l == Label.M)

        for lam in self.lambdas:
            prob_benign = lam / (1 + lam)

            for i in range(self.batch_loops):
                if i % int(self.batch_loops / 10) == 0:
                    print("*", end="")

                self.trainer.batch_step(
                    benign_query_profiles,
                    malicious_query_profiles,
                    prob_benign
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
