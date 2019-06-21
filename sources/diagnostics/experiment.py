from random import seed

from dataset import Dataset
from evaluate import accuracy
from features.param_creator_base import ParamFeaturesComposer
from features.param_creator_units import Histogram, Count, TimeEntropy, BinCounter
from models.pytorch.langrange_net import MonteCarloNet
from models.sampling.sampler import TensorSampler, Sampler
from show import ExperimentHelper
from threat_model.histogram_attacker import FGSMAttacker

seed(42)

###########
experiment_filepath: str = False
################

# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments/http_fee_ctu/langrange_net_fgsm_FPR_0.1_adjusted_grad_new_features/"

requests_filepath = "data/trend_micro_full/user_queries.csv"
scores_filepath = "data/trend_micro_full/url_scores.csv"
critical_urls_filepath = "data/trend_micro_full/critical_urls.csv"
experiment_filepath = "../results/experiments/trend_micro_full/langrange_net_fgsm_FPR_0.1/"

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
attacker = FGSMAttacker(
    featurizer,
    dataset.urls,
    {
        "max_attack_cost": 99.0,
        "private_cost_multiplier": 0.05,
        "uncover_cost": 100.0,
    },
    dataset.specificity,
    max_iterations=400,
    change_rate=1.0
)

# attacker = GoodQueriesAttacker(
#     featurizer,
#     dataset.legitimate_queries,
#     {
#         "max_attack_cost": 99.0,
#         "private_cost_multiplier": 0.5,
#         "uncover_cost": 100.0
#     },
#     100
# )

# Models
# model = NaiveBayes(specificity)
# model = DeepNet(specificity)
# model = AdvDeepNet(specificity)
# model = SVM(specificity, kernel="lin")
# model = SVM(specificity, kernel="rbf")
# model = TorchDeepNet(featurizer) # probably not finished
model = MonteCarloNet(
    featurizer,
    attacker,
    batch_loops=100,
    lambda_init=1.0,
    batch_size=100,
    fp_threshold=0.001,
    lr=0.01,
    lambda_lr=100.0
)

#################
# Old type training in which the model is fitted on attacks that were successful at the previous iteration
#
# qps_trn = dataset.qps_trn
# labels_trn = dataset.labels_trn
#
# qps_trn_mal = [qp for qp, label in zip(qps_trn, labels_trn) if label == Label.M]
# qps_trn_ben = [qp for qp, label in zip(qps_trn, labels_trn) if label == Label.B]
# labels_trn_ben = [Label.B] * len(qps_trn_ben)
#
# for i in range(0, 200):
#     model.fit(qps_trn, labels_trn)
#     att_res = attacker.attack(model, qps_trn_mal)
#
#     undected_qps = att_res.get_undetected_query_profiles()
#
#     qps_trn = qps_trn_ben + undected_qps
#     labels_trn = [Label.B] * len(qps_trn_ben) + [Label.M] * len(undected_qps)
#
#     log("Iter: %d" % i)
#     log("Total attacks number: %d" % att_res.total_attacks_number)
#     log("No Attack Success: %5.2f%%" % (100 * att_res.no_obfuscation_success_rate))
#     log("Attack Success: %5.2f%%" % (100 * att_res.attack_success_rate))
#     log("Mean attack iter: %5.2f" % att_res.mean_attack_step)
#
#     log("Accuracy on Benign Data: %5.2f %% \n" % (accuracy(model, qps_trn_ben, labels_trn_ben) * 100))
#
#     explain_model(model, qps_trn, labels_trn, title="Retraining Iteration i=%d" % i)

#####

benign_samples = TensorSampler([featurizer.make_features(qp, use_torch=True) for qp in dataset.qps_trn_ben])
malicious_samples = Sampler(dataset.qps_trn_mal)

helper = ExperimentHelper(
    featurizer,
    dataset.qps_trn,
    save_folder=experiment_filepath
)

helper.explain_model(
    model,
    dataset.qps_trn,
    dataset.labels_trn,
    title="Initial setting"
)

lambdas = []
att_results_trn = []
att_results_tst = []
benign_accuracy_trn = []
benign_accuracy_tst = []

for i in range(0, 100):
    helper.log("=================================================")
    helper.log("Epoch: %d" % i)

    helper.log("Fitting.")
    model.fit_samplers(benign_samples, malicious_samples)

    # helper.log("TRN Attacking.")
    # att_res_trn = attacker.attack(model, dataset.qps_trn_mal)
    accuracy_trn = accuracy(model, dataset.qps_trn_ben, dataset.labels_trn_ben)
    lam = float(model.model.lam)
    #
    # helper.log("Accuracy on Benign Data: %5.2f%%" % (accuracy_trn * 100))
    # helper.log("Detection Rate: %5.2f%%" % (100 * (1 - att_res_trn.attack_success_rate)))
    #
    # helper.log("Lambda: %5.2f" % lam)
    # helper.log("p(B): %7.4f" % (lam / (1 + lam)))
    #
    # helper.log("Total benign samples number: %d" % len(dataset.qps_trn_ben))
    # helper.log("Total attacks number: %d" % att_res_trn.total_attacks_number)
    #
    # helper.log("No Attack Rate %5.2f%%" % (100 * att_res_trn.no_attack_rate))
    # helper.log("Attack Success: %5.2f%%" % (100 * att_res_trn.attack_success_rate))
    # helper.log("No Obfuscation Success: %5.2f%%" % (100 * att_res_trn.no_obfuscation_success_rate))
    # helper.log("Mean attack iter: %5.2f" % att_res_trn.mean_attack_step)

    helper.log()
    helper.log("lambda: %5.2f" % lam)
    helper.log("TST RESULTS")

    helper.log("TST Attacking.")
    att_res_tst = attacker.attack(model, dataset.qps_tst_mal)

    accuracy_tst = accuracy(model, dataset.qps_tst_ben, dataset.labels_tst_ben)

    helper.log("Accuracy on Benign Data: %5.2f %%" % (accuracy_tst * 100))
    helper.log("Detection Rate: %5.2f%%" % (100 * (1 - att_res_tst.attack_success_rate)))

    helper.log("Total benign samples number: %d" % len(dataset.qps_tst_ben))
    helper.log("Total attacks number: %d" % att_res_tst.total_attacks_number)

    helper.log("NAR %5.2f%%" % (100 * att_res_tst.no_attack_rate))
    helper.log("SCR: %5.2f%%" % (100 * att_res_tst.attack_success_rate))
    helper.log("OBR: %5.2f%%" % (100 * (att_res_tst.attack_success_rate / (1 - att_res_tst.no_attack_rate + 0.00000001))))
    helper.log("No Obfuscation Success: %5.2f%%" % (100 * att_res_tst.no_obfuscation_success_rate))
    helper.log("MAL: %5.2f" % att_res_tst.mean_attack_step)

    lambdas.append(lam)
    # att_results_trn.append(att_res_trn)
    att_results_tst.append(att_res_tst)
    benign_accuracy_trn.append(accuracy_trn)
    benign_accuracy_tst.append(accuracy_tst)

    helper.save_data({
        "lambdas": lambdas,
        "att_res_trn": att_results_trn,
        "att_res_tst": att_results_tst,
        "benign_accuracy_trn": benign_accuracy_trn,
        "benign_accuracy_tst": benign_accuracy_tst,
    })

    # Saving and showing pictures occurs every N loops
    if ((i + 1) % 1) == 0:
        helper.save_model(model, i)
        helper.save_model(model)  # rewrite final

        helper.explain_model(
            model,
            dataset.qps_tst_ben + att_res_tst.get_query_profiles(),
            dataset.labels_tst_ben + dataset.labels_tst_mal,
            title="Training Epoch i=%d" % i
        )

    helper.log("=================================================")
