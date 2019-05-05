from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader, TensorDataset

from features.features_creator import SpecificityHistogram, FeaturesComposer
from models.base import Label, ModelBase
from models.query_profiles import QueryProfile
from probability import Probability

torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()

        self.dense1 = nn.Linear(n_features, n_features * 2)
        self.dense2 = nn.Linear(n_features * 2, n_features * 2)
        # self.dense3 = nn.Linear(n_features * 2, n_features * 2)
        # self.dense4 = nn.Linear(n_features * 2, n_features * 2)
        # self.dense5 = nn.Linear(n_features * 2, n_features * 2)
        # self.dense6 = nn.Linear(n_features * 2, n_features * 2)
        # self.dense7 = nn.Linear(n_features * 2, n_features * 2)
        # self.dense8 = nn.Linear(n_features * 2, n_features * 2)

        self.dense_final = nn.Linear(n_features * 2, 2)

    def forward(self, x):
        x = func.relu(self.dense1(x))
        x = func.relu(self.dense2(x))
        # x = func.relu(self.dense3(x))
        # x = func.relu(self.dense4(x))
        # x = func.relu(self.dense5(x))
        # x = func.relu(self.dense6(x))
        # x = func.relu(self.dense7(x))
        # x = func.relu(self.dense8(x))

        x = func.relu(self.dense_final(x))
        x = func.dropout(x, training=self.training)

        return x


class DeepNet(ModelBase):
    def __init__(self, specificity: Probability, n_bins=3):
        self.featurizer = FeaturesComposer([
            SpecificityHistogram(specificity, n_bins),
            # EntriesCount()
        ])

        self.model = Net(self.featurizer.n_features)
        self.model.double()
        #     model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

        self.criterion = torch.nn.CrossEntropyLoss()

    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        self.model.training = True

        learning_rate = 0.001
        epochs = 500
        batch_size = 32

        X, y = self.featurizer.make_features(query_profiles, labels)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for i, (X_i, y_i) in enumerate(train_loader):
                y_i_est = self.model(X_i)
                loss = self.criterion(y_i_est, y_i)

                self.model.zero_grad()
                loss.backward()

                # actual weights update
                for param in self.model.parameters():
                    param.data.sub_(param.grad.data * learning_rate)  # inplace

    def predict(self, query_profiles: List[QueryProfile]):
        self.model.training = False

        X = self.featurizer.make_features(query_profiles)
        pred_one_hot = self.model(torch.from_numpy(X))
        _, pred = torch.max(pred_one_hot, 1)

        return [Label.from_int(i) for i in pred]

    def predict_raw(self, X):
        self.model.training = False

        pred_one_hot = self.model(torch.from_numpy(X))
        y = func.softmax(pred_one_hot, dim=1)

        return y.detach().numpy()[:, 1]

    def input_gradient(self, query_profiles: List[QueryProfile], labels: List[Label]):
        self.model.training = False

        X, y = self.featurizer.make_features(query_profiles, labels)
        X_torch, y_torch = torch.from_numpy(X), torch.from_numpy(y)
        X_torch.requires_grad = True

        pred = self.model(X_torch)
        loss = self.criterion(pred, y_torch)

        self.model.zero_grad()
        loss.backward()

        return X_torch.grad.data.detach().numpy(), y
