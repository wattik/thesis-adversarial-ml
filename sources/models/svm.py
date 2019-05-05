from typing import List

from sklearn.svm import LinearSVC, SVC

from features.features_creator import SpecificityHistogram, FeaturesComposer, EntriesCount
from models.base import Label, ModelBase
from models.query_profiles import QueryProfile
from probability import Probability


class SVM(ModelBase):
    def __init__(self, specificity: Probability, n_bins=3, kernel="lin"):
        self.featurizer = FeaturesComposer([
            SpecificityHistogram(specificity, n_bins),
            # EntriesCount()
        ])

        if kernel == "lin":
            self.model = LinearSVC()
        else:
            self.model = SVC(gamma="scale")

    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        X, y = self.featurizer.make_features(query_profiles, labels)
        self.model.fit(X, y)

    def predict(self, query_profiles: List[QueryProfile]):
        X = self.featurizer.make_features(query_profiles)
        pred = self.model.predict(X)

        return [Label.from_int(i) for i in pred]
