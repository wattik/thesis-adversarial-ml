import os
import pickle
from random import seed

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from show import ExperimentHelper

seed(42)

################
experiment_root = "../../results/experiments/"
dataset = "trend_micro_full"
experiment = "langrange_net_fgsm_FPR_1"

#################
experiment_filepath = os.path.join(experiment_root, dataset, experiment)

# UnPickling
with open(os.path.join(experiment_filepath, "lambdas.pickle"), "rb") as file:
    lambdas = np.array(pickle.load(file))

with open(os.path.join(experiment_filepath, "benign_accuracy_trn.pickle"), "rb") as file:
    benign_accuracy_trn = pickle.load(file)

with open(os.path.join(experiment_filepath, "benign_accuracy_tst.pickle"), "rb") as file:
    benign_accuracy_tst = pickle.load(file)

with open(os.path.join(experiment_filepath, "att_res_tst.pickle"), "rb") as file:
    att_res_tst = pickle.load(file)


model, _, _ = ExperimentHelper.load(experiment_filepath)
threshold = model.trainer.fp_thresh

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(15, 5), dpi=150, constrained_layout=False)
f.suptitle("Results on Eval Set During Training")


p_m = 1 / (1 + lambdas)
nars = [att.no_attack_rate for att in att_res_tst]
scrs = [att.attack_success_rate for att in att_res_tst]
obrs = [att.attack_success_rate / (1 - att.no_attack_rate + 0.00000001) for att in att_res_tst]
fprs = [1 - acc for acc in benign_accuracy_tst]

ax1.plot(p_m, "r-", label="P(M)")

ax1.plot([0, len(lambdas) - 1], [threshold, threshold], "b--")
ax1.plot(fprs, "b-", label="FPR")

ax2.plot(nars, "g-", label="NAR")
ax2.plot(obrs, "r-", label="OBR")
ax2.plot(scrs, "b-", label="SAR")

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
print("Satisfactory models:")

for i, (fpr, scr, nar) in enumerate(zip(fprs, scrs, nars)):
    if fpr <= threshold:
        print("%d" % i)
        print("FPR: %5.4f%%" % (100 * fpr))
        print("SCR: %5.2f%%" % (100 * scr))
        print("NAR: %5.2f%%" % (100 * nar))
        print()
