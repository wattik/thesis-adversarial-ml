from random import seed
import os

from dataset import Dataset
from evaluate import accuracy
# from features.features_creator import FeaturesComposer, SpecificityHistogram, EntriesCount
from features.param_creator_base import ParamFeaturesComposer
from features.param_creator_units import Histogram, Count, TimeEntropy
from models.pytorch.langrange_net import MonteCarloNet
from models.sampling.sampler import TensorSampler, Sampler
from show import ExperimentHelper
from threat_model.histogram_attacker import HistogramFGSMAttacker

seed(42)

###########
experiment_filepath: str = False
################
#
requests_filepath = "data/http_fee_ctu/user_queries.csv"
scores_filepath = "data/http_fee_ctu/url_scores.csv"
critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
experiment_filepath = "../results/experiments/http_fee_ctu/fgsm_more_features/"

# requests_filepath = "data/trend_micro_full/user_queries.csv"
# scores_filepath = "data/trend_micro_full/url_scores.csv"
# critical_urls_filepath = "data/trend_micro_full/critical_urls.csv"
# experiment_filepath = "../results/experiments/trend_micro_full/fgsm_complex_net/"

# requests_filepath = "data/user_queries.csv"
# scores_filepath = "data/url_scores.csv"
# critical_urls_filepath = "data/critical_urls.csv"

##########

dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)

# featurizer = ParamFeaturesComposer([
#     Histogram(dataset.specificity, 3),
#     Count(),
#     TimeEntropy(24)
# ])
#
# attacker = HistogramFGSMAttacker(
#     featurizer,
#     dataset.urls,
#     {
#         "max_attack_cost": 100.0,
#         "private_cost_multiplier": 0.05,
#         "uncover_cost": 100.0
#     },
#     dataset.specificity,
#     max_iterations=1000,
#     change_rate=1.0
# )

# model = NaiveBayes(specificity)
# model = DeepNet(specificity)
# model = AdvDeepNet(specificity)
# model = SVM(specificity, kernel="lin")
# model = SVM(specificity, kernel="rbf")
# model = TorchDeepNet(featurizer) # probably not finished
# model = MonteCarloNet(
#     featurizer,
#     attacker,
#     batch_loops=100,
#     lambda_init=99.0,
#     batch_size=32,
#     fp_threshold=0.001,
#     lr=0.1,
#     lambda_lr = 10.0
# )

#################
model, featurizer, attacker = ExperimentHelper.load(experiment_filepath)

benign_samples = TensorSampler([featurizer.make_features(qp, use_torch=True) for qp in dataset.qps_trn_ben])
malicious_samples = Sampler(dataset.qps_trn_mal)

helper = ExperimentHelper(
    featurizer,
    dataset.qps_trn
)

helper.explain_model(
    model,
    dataset.qps_trn,
    dataset.labels_trn,
    title="Initial setting"
)

results = attacker.attack(model, dataset.qps_trn_mal)

helper.log("Lambda: %5.2f" % float(model.model.lam))
helper.log("Total attacks number: %d" % results.total_attacks_number)
helper.log("No Obfuscation Success: %5.2f%%" % (100 * results.no_obfuscation_success_rate))
helper.log("Attack Success: %5.2f%%" % (100 * results.attack_success_rate))
helper.log("Mean attack iter: %5.2f" % results.mean_attack_step)

helper.log(
    "Accuracy on Benign Data: %5.2f %% \n" % (accuracy(model, dataset.qps_trn_ben, dataset.labels_trn_ben) * 100)
)

helper.explain_model(
    model,
    dataset.qps_trn_ben + results.get_query_profiles(),
    dataset.labels_trn_ben + dataset.labels_trn_mal,
    title="Attack Results"
)

#######

helper.log()
helper.log("TST RESULTS")

results = attacker.attack(model, dataset.qps_tst_mal)

helper.log("Benign trn data ratio: %5.2f %%" % (
        len(dataset.labels_trn_ben) / len(dataset.labels_trn) * 100
))

acc = accuracy(model, dataset.qps_tst_ben, dataset.labels_tst_ben)
helper.log("Accuracy on Benign Data: %5.2f %% \n" % (acc * 100))
helper.log("Total attacks number: %d" % results.total_attacks_number)
helper.log("No Obfuscation Success: %5.2f%%" % (100 * results.no_obfuscation_success_rate))
helper.log("Attack Success: %5.2f%%" % (100 * results.attack_success_rate))
helper.log("Mean success. attack iter: %5.2f" % results.mean_attack_step)
helper.log()

for qp in results.get_query_profiles():
    helper.log(qp)
    helper.log()
