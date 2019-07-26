import os
import pickle
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from models.base import Label
from models.query_profiles import NoQueriesProfile


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

        x_min, x_max = X_proj[:, 0].min(), X_proj[:, 0].max()
        y_min, y_max = X_proj[:, 1].min(), X_proj[:, 1].max()
        h = 50
        A_proj = np.array(list(product(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))))
        A = self.pca.inverse_transform(A_proj)

        if hasattr(model, "predict_raw"):
            y_prob = model.predict_raw(A)
        elif hasattr(model, "predict_prob"):
            input = torch.from_numpy(A).float()
            y_prob = model.predict_prob(input, Label.M).detach().numpy()
        else:
            raise ValueError("Model type not supported.")

        levels = np.linspace(0, 1, 10)
        plt.tricontour(A_proj[:, 0], A_proj[:, 1], y_prob, levels=levels, linewidths=1.5, cmap="coolwarm")

        no_act_idx = np.array([isinstance(qp, NoQueriesProfile) for qp in query_profiles])
        ben_idx = np.array([l == Label.B for l in labels])
        att_idx = ~no_act_idx & ~ben_idx
        plt.scatter(X_proj[ben_idx, 0], X_proj[ben_idx, 1], c="#3b4cc0", alpha=0.5, label="Benign")
        plt.scatter(X_proj[no_act_idx, 0], X_proj[no_act_idx, 1], c="black", alpha=0.5, label="Malicious No-Activity")
        plt.scatter(X_proj[att_idx, 0], X_proj[att_idx, 1], c="#b40426", alpha=0.5, label="Malicious Attack")

        plt.legend()
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

    def save_attacks(self, attacks, save_folder, version=""):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        with open(os.path.join(save_folder, "attack%s.pickle" % version), "wb") as file:
            pickle.dump(attacks, file)

    @staticmethod
    def load(save_folder, version="final"):
        version = str(version)

        with open(os.path.join(save_folder, "model-%s.torch" % version), "rb") as file:
            model = pickle.load(file)
            featurizer = model.featurizer
            attacker = model.attacker if hasattr(model, "attacker") else None

        return model, featurizer, attacker
