import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from models.base import Label


class FeatureSpaceShower:
    def __init__(self, featurizer, query_profiles, save_folder=None):
        if save_folder:
            if not os.path.exists(save_folder):
                raise Exception(save_folder)

        self.save_folder = save_folder
        self.featurizer = featurizer

        self.pca = PCA(n_components=2)
        self.pca.fit(self.featurizer.make_features(query_profiles))

    def explain_model(self, model, query_profiles, labels, title=""):
        X, y = self.featurizer.make_features(query_profiles, labels)
        X_proj = self.pca.transform(X)

        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap="coolwarm", alpha=0.5)
        colorbar = plt.colorbar()

        if hasattr(model, "predict_raw"):
            y_prob = model.predict_raw(X)
        elif hasattr(model, "predict_prob"):
            input = torch.from_numpy(X).float()
            y_prob = model.predict_prob(input, Label.M).detach().numpy()
        else:
            raise ValueError("Model type not supported.")

        levels = np.linspace(0, 1, 20)
        contour = plt.tricontour(X_proj[:, 0], X_proj[:, 1], y_prob, levels=levels, linewidths=1.5, cmap="coolwarm")

        colorbar.add_lines(contour)

        plt.title(title)

        if self.save_folder:
            plt.savefig(os.path.join(self.save_folder, title + ".png"))

        plt.show()
        plt.clf()
