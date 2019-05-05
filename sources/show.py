import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from models.base import Label


class ExperimentHelper:
    def __init__(self, featurizer, query_profiles, save_folder=None):
        if save_folder:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            self.std_out = open(os.path.join(save_folder, "std.out.txt"), "w", buffering=1)

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
            plt.savefig(os.path.join(self.save_folder, title + ".png"), dpi=500)

        plt.show()
        plt.clf()

    def log(self, t: str = ""):
        t = str(t)
        print(t)
        if self.save_folder:
            s = datetime.now().isoformat()
            s += ": "
            s += t
            s += "\n"
            self.std_out.write(s)

    def save_model(self, model, version="final"):
        version = str(version)

        if self.save_folder:
            with open(os.path.join(self.save_folder, "model-%s.torch" % version), "wb") as file:
                pickle.dump(model, file)
        else:
            self.log("Model not saved. (Missing experiment folder)")

    def save_data(self, data):
        if self.save_folder:
            for name, d in data.items():
                with open(os.path.join(self.save_folder, "%s.pickle" % name), "wb") as file:
                    pickle.dump(d, file)
        else:
            self.log("Data not saved. (Missing experiment folder)")

    @staticmethod
    def load(save_folder, version="final"):
        version = str(version)

        if save_folder:
            with open(os.path.join(save_folder, "model-%s.torch" % version), "rb") as file:
                model = pickle.load(file)
            return model, model.featurizer, model.attacker
        else:
            print("Model not loaded. (Missing experiment folder)")
