import os
import pickle
from random import seed
from matplotlib import pyplot as plt
import numpy as np
# from features.features_creator import FeaturesComposer, SpecificityHistogram, EntriesCount

seed(42)

################
#
# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments/http_fee_ctu/fgsm_more_features/"

requests_filepath = "data/trend_micro_full/user_queries.csv"
scores_filepath = "data/trend_micro_full/url_scores.csv"
critical_urls_filepath = "data/trend_micro_full/critical_urls.csv"
# experiment_filepath = "../results/experiments/trend_micro_full/langrange_net_fgsm_small_input_space/"

# requests_filepath = "data/user_queries.csv"
# scores_filepath = "data/url_scores.csv"
# critical_urls_filepath = "data/critical_urls.csv"

#################
experiment_root = "../../results/experiments/"
experiment = "trend_micro_full/langrange_net_fgsm_small_input_space"

experiment_filepath = os.path.join(experiment_root, experiment)

# UnPickling
with open(os.path.join(experiment_filepath, "lambdas.pickle"), "rb") as file:
    lambdas = np.array(pickle.load(file))

with open(os.path.join(experiment_filepath, "benign_accuracy_trn.pickle"), "rb") as file:
    benign_accuracy_trn = pickle.load(file)

with open(os.path.join(experiment_filepath, "benign_accuracy_tst.pickle"), "rb") as file:
    benign_accuracy_tst = pickle.load(file)

with open(os.path.join(experiment_filepath, "att_res_trn.pickle"), "rb") as file:
    att_res_trn = pickle.load(file)

with open(os.path.join(experiment_filepath, "att_res_tst.pickle"), "rb") as file:
    att_res_tst = pickle.load(file)

p_b = lambdas / (1 + lambdas)
plt.plot(p_b, "r-", label="P(B)")

plt.plot(benign_accuracy_tst, "b-", label="ACC")
plt.plot([att.no_attack_rate for att in att_res_tst], "p-", label="NOA")
plt.plot([1 - att.attack_success_rate for att in att_res_tst], "g--", label="DET")

plt.xlabel("Epochs")

plt.legend()
plt.show()
