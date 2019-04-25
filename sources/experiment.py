from random import seed

from dataset import Dataset
from evaluate import accuracy
from features.creator import FeaturesComposer, SpecificityHistogram
from models.pytorch.langrange_net import MonteCarloNet
from show import FeatureSpaceShower
from threat_model.histogram_attacker import HistogramAttacker

seed(42)

###########

# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments/http_fee_ctu/adv_att_no_att/"

requests_filepath = "data/trend_micro_full/user_queries.csv"
scores_filepath = "data/trend_micro_full/url_scores.csv"
critical_urls_filepath = "data/trend_micro_full/critical_urls.csv"
experiment_filepath = "../results/experiments/trend_micro_full/adv_att_no_att/"

# requests_filepath = "data/user_queries.csv"
# scores_filepath = "data/url_scores.csv"
# critical_urls_filepath = "data/critical_urls.csv"

##########

dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)

specificity = dataset.specificity
legitimate_queries = dataset.legitimate_queries

featurizer = FeaturesComposer([
    SpecificityHistogram(specificity, 3),
    # EntriesCount(),
    # RandomVector(specificity, 3)
])

attacker = HistogramAttacker(
    featurizer,
    dataset.urls,
    {
        "max_attack_cost": 70.0,
        "private_cost_multiplier": 0.5,
        "uncover_cost": 100.0
    },
    specificity,
    max_iterations=10,
    change_rate=10.0
)

# model = NaiveBayes(specificity)
# model = DeepNet(specificity)
# model = AdvDeepNet(specificity)
# model = SVM(specificity, kernel="lin")
# model = SVM(specificity, kernel="rbf")
# model = TorchDeepNet(featurizer) # probably not finished
model = MonteCarloNet(featurizer, attacker, batch_loops=100, lambda_init=9.0)

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
#     results = attacker.attack(model, qps_trn_mal)
#
#     undected_qps = results.get_undetected_query_profiles()
#
#     qps_trn = qps_trn_ben + undected_qps
#     labels_trn = [Label.B] * len(qps_trn_ben) + [Label.M] * len(undected_qps)
#
#     print("Iter: %d" % i)
#     print("Total attacks number: %d" % results.total_attacks_number)
#     print("No Attack Success: %5.2f%%" % (100 * results.no_attack_success_rate))
#     print("Attack Success: %5.2f%%" % (100 * results.attack_success_rate))
#     print("Mean attack iter: %5.2f" % results.mean_attack_step)
#
#     print("Accuracy on Benign Data: %5.2f %% \n" % (accuracy(model, qps_trn_ben, labels_trn_ben) * 100))
#
#     explain_model(model, qps_trn, labels_trn, title="Retraining Iteration i=%d" % i)

#####

plotter = FeatureSpaceShower(featurizer, dataset.qps_trn, save_folder=experiment_filepath)

plotter.explain_model(
    model,
    dataset.qps_trn,
    dataset.labels_trn,
    title="Initial setting"
)

for i in range(0, 100):
    print("Fitting.")
    model.fit(dataset.qps_trn, dataset.labels_trn)

    print("Attacking.")
    results = attacker.attack(model, dataset.qps_trn_mal)

    print("Iter: %d" % i)
    print("Lambda: %5.2f" % float(model.model.lam))
    print("Total attacks number: %d" % results.total_attacks_number)
    print("No Attack Success: %5.2f%%" % (100 * results.no_attack_success_rate))
    print("Attack Success: %5.2f%%" % (100 * results.attack_success_rate))
    print("Mean attack iter: %5.2f" % results.mean_attack_step)

    print("Accuracy on Benign Data: %5.2f %% \n" % (accuracy(model, dataset.qps_trn_ben, dataset.labels_trn_ben) * 100))

    plotter.explain_model(
        model,
        dataset.qps_trn_ben + results.get_query_profiles(),
        dataset.labels_trn_ben + dataset.labels_trn_mal,
        title="Training epoch i=%d" % i
    )
    print("Iter END.")

#######

print()
print("TST RESULTS")

results = attacker.attack(model, dataset.qps_tst_mal)

print("Benign trn data ratio: %5.2f %%" % (
        len(dataset.labels_trn_ben) / len(dataset.labels_trn) * 100
))

acc = accuracy(model, dataset.qps_tst_ben, dataset.labels_tst_ben)
print("Accuracy on Benign Data: %5.2f %% \n" % (acc * 100))
print("Total attacks number: %d" % results.total_attacks_number)
print("No Attack Success: %5.2f%%" % (100 * results.no_attack_success_rate))
print("Attack Success: %5.2f%%" % (100 * results.attack_success_rate))
print("Mean success. attack iter: %5.2f" % results.mean_attack_step)

