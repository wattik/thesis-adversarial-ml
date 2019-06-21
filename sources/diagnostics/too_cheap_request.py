import os
from random import seed
from time import time

from dataset import Dataset
from evaluate import accuracy
from show import ExperimentHelper
from threat_model.histogram_attacker import FGSMAttacker


def evaluate(model, attacker, featurizer, name=""):
    helper = ExperimentHelper(
        featurizer,
        dataset.qps_trn
    )

    helper.log("TST RESULTS")

    att_res = attacker.attack(model, dataset.qps_tst_mal)
    accuracy_tst = accuracy(model, dataset.qps_tst_ben, dataset.labels_tst_ben)

    helper.log("Accuracy on Benign Data: %5.2f %%" % (accuracy_tst * 100))
    helper.log("Detection Rate: %5.2f%%" % (100 * (1 - att_res.attack_success_rate)))

    helper.log("Total benign samples number: %d" % len(dataset.qps_tst_ben))
    helper.log("Total attacks number: %d" % att_res.total_attacks_number)

    helper.log("No Attack Rate %5.2f%%" % (100 * att_res.no_attack_rate))
    helper.log("Attack Success: %5.2f%%" % (100 * att_res.attack_success_rate))
    helper.log("No Obfuscation Success: %5.2f%%" % (100 * att_res.no_obfuscation_success_rate))
    helper.log("Mean attack iter: %5.2f" % att_res.mean_attack_step)

    helper.explain_model(
        model,
        dataset.qps_tst_ben + att_res.get_query_profiles(),
        dataset.labels_tst_ben + dataset.labels_tst_mal,
        title="Test Attack Results" + name
    )


seed(42)

################
#
# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments/http_fee_ctu/fgsm_more_features/"

requests_filepath = "../data/trend_micro_full/user_queries.csv"
scores_filepath = "../data/trend_micro_full/url_scores.csv"
critical_urls_filepath = "../data/trend_micro_full/critical_urls.csv"
dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)

# dataset_path = "../data/trend_micro_full/"
# dataset.save(dataset_path)
# dataset = Dataset.load(dataset_path)

# requests_filepath = "data/user_queries.csv"
# scores_filepath = "data/url_scores.csv"
# critical_urls_filepath = "data/critical_urls.csv"

#################

experiment_root = "../../results/experiments/"
experiment = "trend_micro_full/langrange_net_fgsm_small_input_space"
experiment_filepath = os.path.join(experiment_root, experiment)

#################
model, featurizer, attacker_ORIG = ExperimentHelper.load(experiment_filepath)

attacker_05 = FGSMAttacker(
    featurizer,
    dataset.urls,
    {
        "max_attack_cost": 99.0,
        "private_cost_multiplier": 0.5,
        "uncover_cost": 100.0
    },
    dataset.specificity,
    max_iterations=400,
    change_rate=1.0
)

attacker_005 = FGSMAttacker(
    featurizer,
    dataset.urls,
    {
        "max_attack_cost": 99.0,
        "private_cost_multiplier": 0.05,
        "uncover_cost": 100.0
    },
    dataset.specificity,
    max_iterations=400,
    change_rate=1.0
)

attacker_0005 = FGSMAttacker(
    featurizer,
    dataset.urls,
    {
        "max_attack_cost": 99.0,
        "private_cost_multiplier": 0.005,
        "uncover_cost": 100.0
    },
    dataset.specificity,
    max_iterations=400,
    change_rate=1.0
)

evaluate(model, attacker_05, featurizer, "  - Request Cost 0.5")
evaluate(model, attacker_005, featurizer, "  - Request Cost 0.05")
evaluate(model, attacker_0005, featurizer, "  - Request Cost 0.005")
evaluate(model, attacker_05, featurizer, "  - Request Cost 0.5")
evaluate(model, attacker_05, featurizer, "  - Request Cost 0.5")
evaluate(model, attacker_05, featurizer, "  - Request Cost 0.5")
evaluate(model, attacker_05, featurizer, "  - Request Cost 0.5")