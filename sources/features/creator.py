from functools import reduce
from operator import add
from random import random, seed
from statistics import mean
from typing import List, Union

import numpy as np
import torch

from models.base import Label
from models.query_profiles import QueryProfile
from probability import Probability


class SpecificityHistogram:
    def __init__(self, specificity: Probability, n_bins):
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.specificity = specificity

    @property
    def output_len(self):
        return len(self.bins) - 1

    def __call__(self, query_profile: QueryProfile):
        scores = list(map(lambda x: self.specificity(x), query_profile.queries))
        hist, _ = np.histogram(scores, self.bins)
        if np.sum(hist) != 0:
            hist = (hist / np.sum(hist))
        return list(hist)


class RandomVector:
    def __init__(self, specificity: Probability, n_dims):
        self.specificity = specificity
        self.n_dims = n_dims

    @property
    def output_len(self):
        return self.n_dims

    def __call__(self, query_profile):
        seed(hash(tuple(query_profile.queries)))

        if mean(map(lambda x: self.specificity(x), query_profile.queries)) < 0.5:
            return [random() + 1.5 for _ in range(self.n_dims)]
        else:
            return [random() for _ in range(self.n_dims)]


class EntriesCount:
    @property
    def output_len(self):
        return 1

    def __call__(self, query_profile: QueryProfile):
        return [np.log(len(query_profile.queries))]


class FeaturesComposer:
    def __init__(self, features_creators):
        self.features_creators = features_creators

    @property
    def n_features(self):
        return sum([f.output_len for f in self.features_creators])

    def make_features(
            self,
            query_profiles: Union[List[QueryProfile], QueryProfile],
            labels: Union[List[Label], Label] = None,
            use_torch=False,
    ):
        if isinstance(query_profiles, QueryProfile):
            query_profiles = [query_profiles]

        if isinstance(labels, Label):
            labels = [labels]

        # actual feature composing

        features = []
        for qp in query_profiles:
            features_list = (f(qp) for f in self.features_creators)
            features.append(list(reduce(add, features_list)))

        if use_torch:
            X = torch.tensor(features, dtype=torch.float)
        else:
            X = np.array(features)

        res = X

        if labels:
            if use_torch:
                y = torch.tensor([label.to_int() for label in labels])
            else:
                y = np.array([label.to_int() for label in labels])

            res = (X, y)

        return res
