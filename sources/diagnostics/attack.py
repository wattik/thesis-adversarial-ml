from random import seed

from dataset import Dataset
from evaluate import accuracy
# from features.features_creator import FeaturesComposer, SpecificityHistogram, EntriesCount
from show import ExperimentHelper
from threat_model.good_queries_attacker import GoodQueriesAttacker
from threat_model.histogram_attacker import FGSMAttacker

seed(42)

###########
experiment_filepath: str = False
################
#
# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments/http_fee_ctu/fgsm_more_features/"

requests_filepath = "data/trend_micro_full/user_queries.csv"
scores_filepath = "data/trend_micro_full/url_scores.csv"
critical_urls_filepath = "data/trend_micro_full/critical_urls.csv"
experiment_filepath = "../results/experiments/trend_micro_full/langrange_net_fgsm/"

# requests_filepath = "data/user_queries.csv"
# scores_filepath = "data/url_scores.csv"
# critical_urls_filepath = "data/critical_urls.csv"

##########

dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)

#################
model, featurizer, attacker = ExperimentHelper.load(experiment_filepath)

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

attacker_GQAT = GoodQueriesAttacker(
    featurizer,
    dataset.legitimate_queries,
    {
        "max_attack_cost": 99.0,
        "private_cost_multiplier": 0.5,
        "uncover_cost": 100.0
    },
    100
)

# benign_samples = TensorSampler([featurizer.make_features(qp, use_torch=True) for qp in dataset.qps_trn_ben])
# malicious_samples = Sampler(dataset.qps_trn_mal)

helper = ExperimentHelper(
    featurizer,
    dataset.qps_trn
)

# helper.explain_model(
#     model,
#     dataset.qps_trn,
#     dataset.labels_trn,
#     title="Initial setting"
# )

att_res_trn = attacker.attack(model, dataset.qps_trn_mal)
accuracy_trn = accuracy(model, dataset.qps_tst_ben, dataset.labels_tst_ben)
lam = float(model.model.lam)

helper.log("Accuracy on Benign Data: %5.2f%%" % (accuracy_trn * 100))
helper.log("Detection Rate: %5.2f%%" % (100 * (1 - att_res_trn.attack_success_rate)))

helper.log("Lambda: %5.2f" % lam)
helper.log("p(B): %7.4f" % (lam / (1 + lam)))

helper.log("Total benign samples number: %d" % len(dataset.qps_trn_ben))
helper.log("Total attacks number: %d" % att_res_trn.total_attacks_number)

helper.log("No Attack Rate %5.2f%%" % (100 * att_res_trn.no_attack_rate))
helper.log("Attack Success: %5.2f%%" % (100 * att_res_trn.attack_success_rate))
helper.log("No Obfuscation Success: %5.2f%%" % (100 * att_res_trn.no_obfuscation_success_rate))
helper.log("Mean attack iter: %5.2f" % att_res_trn.mean_attack_step)

helper.explain_model(
    model,
    dataset.qps_trn_ben + att_res_trn.get_query_profiles(),
    dataset.labels_trn_ben + dataset.labels_trn_mal,
    title="Attack Results in Training"
)

#######

helper.log()
helper.log("TST RESULTS")

att_res_tst = attacker.attack(model, dataset.qps_tst_mal)
accuracy_tst = accuracy(model, dataset.qps_tst_ben, dataset.labels_tst_ben)

helper.log("Accuracy on Benign Data: %5.2f %%" % (accuracy_tst * 100))
helper.log("Detection Rate: %5.2f%%" % (100 * (1 - att_res_tst.attack_success_rate)))

helper.log("Total benign samples number: %d" % len(dataset.qps_tst_ben))
helper.log("Total attacks number: %d" % att_res_tst.total_attacks_number)

helper.log("No Attack Rate %5.2f%%" % (100 * att_res_tst.no_attack_rate))
helper.log("Attack Success: %5.2f%%" % (100 * att_res_tst.attack_success_rate))
helper.log("No Obfuscation Success: %5.2f%%" % (100 * att_res_tst.no_obfuscation_success_rate))
helper.log("Mean attack iter: %5.2f" % att_res_tst.mean_attack_step)

helper.explain_model(
    model,
    dataset.qps_trn_ben + att_res_tst.get_query_profiles(),
    dataset.labels_trn_ben + dataset.labels_trn_mal,
    title="Attack Results in Testing"
)

for qp in att_res_tst.get_query_profiles()[0:10]:
    helper.log(qp)
    helper.log()
