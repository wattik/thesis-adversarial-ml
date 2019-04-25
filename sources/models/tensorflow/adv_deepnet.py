from typing import List

import tensorflow as tf
from tensorflow import keras

import utils
from features.creator import SpecificityHistogram, FeaturesComposer, EntriesCount
from models.base import Label, ModelBase
from models.query_profiles import QueryProfile
from probability import Probability

tf.enable_eager_execution()
tf.executing_eagerly()

class AdversarialOptimizer(keras.optimizers.Adam):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.delta = 0.1

    def get_updates(self, loss, params):
        x = self.model.inputs[0]
        y = self.model.targets[0]

        dx = keras.backend.gradients(loss, x)[0]

        x_adv = x + tf.multiply(self.delta, dx)

        print(x)
        mal_idx = tf.broadcast_to(y[:, 1], x.shape)
        ben_idx = tf.broadcast_to(y[:, 0], x.shape)

        res = tf.multiply(ben_idx, x) + tf.multiply(mal_idx, x_adv)

        print(res)

        tf.assign(self.model.inputs, res)

        super(AdversarialOptimizer, self).get_updates(loss, params)


class AdvDeepNet(ModelBase):
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

        self.model.compile(optimizer=AdversarialOptimizer(self.model),
                           # optimizer=tf.train.AdamOptimizer(0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, query_profiles: List[QueryProfile], labels: List[Label]):
        X, y = self.featurizer.make_features(query_profiles, labels)
        y_one_hot = utils.to_one_hot(y)
        self.model.fit(X, y_one_hot,
                       epochs=50,
                       batch_size=32,
                       use_multiprocessing=True,
                       )

    def predict(self, query_profiles: List[QueryProfile]):
        X = self.featurizer.make_features(query_profiles)
        pred_one_hot = self.model.predict(X)
        pred = utils.from_one_hot(pred_one_hot)
        return [Label.from_int(i) for i in pred]
