from math import log, inf
from typing import List

from models.base import Label, ModelBase
from models.query_profiles import QueryProfile
from probability import Probability

EPS = 0.001


def logify(p):
    p = max(p, EPS)
    p = min(p, 1 - EPS)
    return log(p)


class NaiveBayes(ModelBase):
    def __init__(self, specificity: Probability):
        self.specificity = specificity

    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        pass

    def predict(self, query_profiles: List[QueryProfile]):
        labels = []
        for query_profile in query_profiles:

            try:
                log_prob_benign = sum(logify(self.specificity(x)) for x in query_profile.queries)
            except ValueError:
                log_prob_benign = -inf

            try:
                log_prob_malicious = sum(logify(1 - self.specificity(x)) for x in query_profile.queries)
            except ValueError:
                log_prob_malicious = -inf

            if log_prob_benign > log_prob_malicious:
                labels.append(Label.B)
            else:
                labels.append(Label.M)

        return labels
