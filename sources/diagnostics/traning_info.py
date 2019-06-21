import os
import pickle
from random import seed

import numpy as np
from matplotlib import pyplot as plt
# from features.features_creator import FeaturesComposer, SpecificityHistogram, EntriesCount
from matplotlib.ticker import PercentFormatter

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
dataset = "trend_micro_full"
experiment = "langrange_net_fgsm_FPR_0.01:from_0.1"

experiment_filepath = os.path.join(experiment_root, dataset, experiment)

# UnPickling
with open(os.path.join(experiment_filepath, "lambdas.pickle"), "rb") as file:
    lambdas = np.array(pickle.load(file))

with open(os.path.join(experiment_filepath, "benign_accuracy_trn.pickle"), "rb") as file:
    benign_accuracy_trn = pickle.load(file)

with open(os.path.join(experiment_filepath, "benign_accuracy_tst.pickle"), "rb") as file:
    benign_accuracy_tst = pickle.load(file)

# with open(os.path.join(experiment_filepath, "att_res_trn.pickle"), "rb") as file:
#     att_res_trn = pickle.load(file)

with open(os.path.join(experiment_filepath, "att_res_tst.pickle"), "rb") as file:
    att_res_tst = pickle.load(file)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(15, 5), dpi=150, constrained_layout=False)
f.suptitle("Results on Eval Set During Training")

threshold = 0.01 / 100

p_m = 1 / (1 + lambdas)
nars = [att.no_attack_rate for att in att_res_tst]
sars = [att.attack_success_rate for att in att_res_tst]
obrs = [att.attack_success_rate / (1 - att.no_attack_rate + 0.00000001) for att in att_res_tst]
fprs = [1 - acc for acc in benign_accuracy_tst]

ax1.plot(p_m, "r-", label="P(M)")

ax1.plot([0, len(lambdas) - 1], [threshold, threshold], "b--")
ax1.plot(fprs, "b-", label="FPR")

ax2.plot(nars, "g-", label="NAR")
ax2.plot(obrs, "r-", label="OBR")
ax2.plot(sars, "b-", label="SAR")

ax3.plot([att.mean_attack_step for att in att_res_tst], "b-", label="MAL")

ax1.set_xlabel("Epochs")
ax1.yaxis.set_major_formatter(PercentFormatter(1))
ax2.set_xlabel("Epochs")
ax2.yaxis.set_major_formatter(PercentFormatter(1))
ax3.set_xlabel("Epochs")

ax1.set_title(' ')  # Hack to avoid title overlapping

ax1.legend()
ax2.legend()
ax3.legend()

plt.show()

######################

for i, (fpr, sar, nar) in enumerate(zip(fprs, sars, nars)):
    if fpr <= 0.05/100:
        print("%d" % i)
        print("FPR: %5.2f%%" % (100 * fpr))
        print("SAR: %5.2f%%" % (100 * sar))
        print("NAR: %5.2f%%" % (100 * nar))
        print()
