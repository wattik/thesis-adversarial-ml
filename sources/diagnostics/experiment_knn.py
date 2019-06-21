from random import seed

from dataset import Dataset
from evaluate import accuracy
# from features.features_creator import FeaturesComposer, SpecificityHistogram, EntriesCount
from features.param_creator_base import ParamFeaturesComposer
from features.param_creator_units import Histogram, Count, TimeEntropy, BinCounter
from models.knn import KNN
from models.sampling.sampler import TensorSampler, Sampler
from show import ExperimentHelper
from threat_model.good_queries_attacker import GoodQueriesAttacker
from threat_model.histogram_attacker import FGSMAttacker

seed(42)

###########
experiment_filepath: str = False
################

# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments/http_fee_ctu/knn_test/"

scores_filepath = "../data/trend_micro_full/url_scores.csv"
requests_filepath = "../data/trend_micro_full/user_queries.csv"
critical_urls_filepath = "../data/trend_micro_full/critical_urls.csv"
experiment_filepath = "../../results/experiments/trend_micro_full/knn_fgsm_FPR_0.01/"

# requests_filepath = "data/user_queries.csv"
# scores_filepath = "data/url_scores.csv"
# critical_urls_filepath = "data/critical_urls.csv"

##########

dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)

# Featurizer
featurizer = ParamFeaturesComposer([
    Histogram(dataset.specificity, 3),
    BinCounter(dataset.specificity, 3),
    Count(),
    TimeEntropy(24)
])
featurizer.fit_normalizer(dataset.qps_trn_ben)

# Attacker
attacker_GRAD = FGSMAttacker(
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

# Model
model = KNN(
    featurizer,
    k=10,
    fp_thresh=0.01/100,
    lr=1.0,
    validation_ratio=0.2,
    alpha_init=180.0,
    max_iter=100
)

benign_samples = TensorSampler([featurizer.make_features(qp, use_torch=True) for qp in dataset.qps_trn_ben])
malicious_samples = Sampler(dataset.qps_trn_mal)

helper = ExperimentHelper(
    featurizer,
    dataset.qps_trn,
    save_folder=experiment_filepath
)

helper.log("Fitting.")
helper.log(experiment_filepath)
model.fit_samplers(benign_samples, malicious_samples)
helper.save_model(model)

helper.log("GRAD")
helper.log()

helper.log("TRN Attacking.")
att_res_trn = attacker_GRAD.attack(model, dataset.qps_trn_mal)
accuracy_trn = accuracy(model, dataset.qps_trn_ben, dataset.labels_trn_ben)

helper.log("Accuracy on Benign Data: %5.2f%%" % (accuracy_trn * 100))
helper.log("Detection Rate: %5.2f%%" % (100 * (1 - att_res_trn.attack_success_rate)))

helper.log("Final alpha: %5.2f" % float(model.alpha))

helper.log("Total benign samples number: %d" % len(dataset.qps_trn_ben))
helper.log("Total attacks number: %d" % att_res_trn.total_attacks_number)

helper.log("No Attack Rate %5.2f%%" % (100 * att_res_trn.no_attack_rate))
helper.log("Attack Success: %5.2f%%" % (100 * att_res_trn.attack_success_rate))
helper.log("No Obfuscation Success: %5.2f%%" % (100 * att_res_trn.no_obfuscation_success_rate))
helper.log("Mean attack iter: %5.2f" % att_res_trn.mean_attack_step)

helper.log()
helper.log("TST RESULTS")

helper.log("TST Attacking.")
att_res_tst = attacker_GRAD.attack(model, dataset.qps_tst_mal)

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
    dataset.qps_trn_ben + att_res_trn.get_query_profiles(),
    dataset.labels_trn_ben + dataset.labels_trn_mal,
    title="Training Results vs. GRAD Attacker."
)

helper.log("GQAT")
helper.log()

helper.log("TRN Attacking.")
att_res_trn = attacker_GQAT.attack(model, dataset.qps_trn_mal)
accuracy_trn = accuracy(model, dataset.qps_trn_ben, dataset.labels_trn_ben)

helper.log("Accuracy on Benign Data: %5.2f%%" % (accuracy_trn * 100))
helper.log("Detection Rate: %5.2f%%" % (100 * (1 - att_res_trn.attack_success_rate)))

helper.log("Final alpha: %5.2f" % float(model.alpha))

helper.log("Total benign samples number: %d" % len(dataset.qps_trn_ben))
helper.log("Total attacks number: %d" % att_res_trn.total_attacks_number)

helper.log("No Attack Rate %5.2f%%" % (100 * att_res_trn.no_attack_rate))
helper.log("Attack Success: %5.2f%%" % (100 * att_res_trn.attack_success_rate))
helper.log("No Obfuscation Success: %5.2f%%" % (100 * att_res_trn.no_obfuscation_success_rate))
helper.log("Mean attack iter: %5.2f" % att_res_trn.mean_attack_step)

helper.log()
helper.log("TST RESULTS")

helper.log("TST Attacking.")
att_res_tst = attacker_GQAT.attack(model, dataset.qps_tst_mal)

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
    dataset.qps_trn_ben + att_res_trn.get_query_profiles(),
    dataset.labels_trn_ben + dataset.labels_trn_mal,
    title="Training Results vs. GQAT Attacker."
)