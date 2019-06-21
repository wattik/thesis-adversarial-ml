from functools import reduce
from operator import add
from typing import List, Union, Callable

import numpy as np
import torch

from models.base import Label
from models.query_profiles import QueryProfile


class SpecificityHistogram:
    def __init__(self, specificity: Callable, n_bins):
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.specificity = specificity

    def __call__(self, query_profile: QueryProfile):
        scores = list(map(lambda x: self.specificity(x), query_profile.queries))
        hist, _ = np.histogram(scores, self.bins)
        if np.sum(hist) != 0:
            hist = (hist / np.sum(hist))
        return list(hist)


class EntriesCount:
    @property
    def output_len(self):
        return 1

    def __call__(self, query_profile: QueryProfile):
        return [len(query_profile.queries)]


class FeaturesComposer:
    def __init__(self, features_creators):
        self.modules = features_creators

    @property
    def n_features(self):
        return sum([f.output_len for f in self.modules])

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
            features_list = (f(qp) for f in self.modules)
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

    def qps_to_bases(self, qps: List[QueryProfile], use_torch=False):
        if not use_torch:
            raise ValueError("Only torch is supported right now.")
