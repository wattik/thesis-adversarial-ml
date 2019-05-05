from typing import List

import tensorflow as tf
from tensorflow import keras

import utils
from features.features_creator import SpecificityHistogram, FeaturesComposer, EntriesCount
from models.base import Label, ModelBase
from models.query_profiles import QueryProfile
from probability import Probability


class DeepNet(ModelBase):
    def __init__(self, specificity: Probability, n_bins=3):
        self.featurizer = FeaturesComposer([
                SpecificityHistogram(specificity, n_bins),
                EntriesCount()
            ])

        self.__make_model()

    def __make_model(self):
        self.model = keras.Sequential([
            keras.layers.Dense(self.featurizer.n_features * 2, activation=tf.nn.relu),
            keras.layers.Dense(self.featurizer.n_features * 2, activation=tf.nn.relu),
            keras.layers.Dense(self.featurizer.n_features * 2, activation=tf.nn.relu),
            keras.layers.Dense(self.featurizer.n_features * 2, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        X, y = self.featurizer.make_features(query_profiles, labels)
        y_one_hot = utils.to_one_hot(y)
        self.model.fit(X, y_one_hot, epochs=50, batch_size=32, verbose=0)

    def predict(self, query_profiles: List[QueryProfile]):
        X = self.featurizer.make_features(query_profiles)
        pred_one_hot = self.model.transform(X)
        pred = utils.from_one_hot(pred_one_hot)
        return [Label.from_int(i) for i in pred]

