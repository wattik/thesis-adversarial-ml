import os
from random import seed

from dataset import Dataset
from evaluate import accuracy
from show import ExperimentHelper
from threat_model.histogram_attacker import FGSMAttacker


def evaluate(model, attacker, dataset):
    att_res_trn = attacker.attack(model, dataset.qps_trn_mal)
    accuracy_trn = accuracy(model, dataset.qps_tst_ben, dataset.labels_tst_ben)
    lam = float(model.model.lam)

    print("Accuracy on Benign Data: %5.2f%%" % (accuracy_trn * 100))
    print("Detection Rate: %5.2f%%" % (100 * (1 - att_res_trn.attack_success_rate)))

    print("Lambda: %5.2f" % lam)
    print("p(B): %7.4f" % (lam / (1 + lam)))

    print("Total benign samples number: %d" % len(dataset.qps_trn_ben))
    print("Total attacks number: %d" % att_res_trn.total_attacks_number)

    print("No Attack Rate %5.2f%%" % (100 * att_res_trn.no_attack_rate))
    print("Attack Success: %5.2f%%" % (100 * att_res_trn.attack_success_rate))
    print("No Obfuscation Success: %5.2f%%" % (100 * att_res_trn.no_obfuscation_success_rate))
    print("Mean attack iter: %5.2f" % att_res_trn.mean_attack_step)

    #######

    print()
    print("TST RESULTS")

    att_res_tst = attacker.attack(model, dataset.qps_tst_mal)
    accuracy_tst = accuracy(model, dataset.qps_tst_ben, dataset.labels_tst_ben)

    print("Accuracy on Benign Data: %5.2f %%" % (accuracy_tst * 100))
    print("Detection Rate: %5.2f%%" % (100 * (1 - att_res_tst.attack_success_rate)))

    print("Total benign samples number: %d" % len(dataset.qps_tst_ben))
    print("Total attacks number: %d" % att_res_tst.total_attacks_number)

    print("NAR %5.2f%%" % (100 * att_res_tst.no_attack_rate))
    print("SCR: %5.2f%%" % (100 * att_res_tst.attack_success_rate))
    print(
        "OBR: %5.2f%%" % (100 * (att_res_tst.attack_success_rate / (1 - att_res_tst.no_attack_rate + 0.00000001))))
    print("No Obfuscation Success: %5.2f%%" % (100 * att_res_tst.no_obfuscation_success_rate))
    print("MAL: %5.2f" % att_res_tst.mean_attack_step)


seed(42)

################
requests_filepath = "../data/trend_micro_full/user_queries.csv"
scores_filepath = "../data/trend_micro_full/url_scores.csv"
critical_urls_filepath = "../data/trend_micro_full/critical_urls.csv"
##########

experiment_root = "../../results/experiments/trend_micro_full/"
experiments_config = [
    # ("langrange_net_fgsm_FPR_0.01_cont_2", "199"),
    # ("langrange_net_fgsm_FPR_0.01_cont_2", "200"),

    ("langrange_net_fgsm_FPR_0.1:b=32_lr=0.001", "83"),
    ("langrange_net_fgsm_FPR_0.1:b=32_lr=0.001", "82"),

    # ("langrange_net_fgsm_FPR_1", "16"),
    # ("langrange_net_fgsm_FPR_1", "15"),
]

dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)

experiments_loaded = [
    ExperimentHelper.load(os.path.join(experiment_root, experiment), version=version)
    for experiment, version in experiments_config
]
#################
for (model, featurizer, _), name in zip(experiments_loaded, experiments_config):
    attacker_GRAD = FGSMAttacker(
        featurizer,
        dataset.urls,
        {
            "max_attack_cost": 100.0,
            "private_cost_multiplier": 0.5,
            "uncover_cost": 100.0
        },
        dataset.specificity,
        max_iterations=400,
        change_rate=1.0
    )

    print(name)

    print("GRAD ATTACKER:")
    evaluate(model, attacker_GRAD, dataset)

    print()
    print()
