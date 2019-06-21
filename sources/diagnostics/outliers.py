from pprint import pprint
from random import seed

from dataset import Dataset
# from features.features_creator import FeaturesComposer, SpecificityHistogram, EntriesCount
from features.param_creator_units import Histogram, BinCounter
from models.base import Label
from show import ExperimentHelper

seed(42)


###########
def print_outliers(threshold):
    x = featurizer.make_features(dataset.qps_tst_ben, use_torch=True)
    pred = model.predict_prob(x, Label.M)

    for user, qp, prob_m in zip(dataset.users_tst_ben, dataset.qps_tst_ben, pred):
        if float(prob_m * 100) < threshold:
            continue

        print("=" * 50)
        print("User: %s" % user)
        print("Probability: %5.2f%%" % float(prob_m * 100))
        print("Scores Histogram:")
        print(histogram.make_features([qp], use_torch=True))
        print("Scores Counts:")
        print(counts.make_features([qp], use_torch=True))

        print("Malicious URLS:")
        pprint([url for url in set(qp.queries) if dataset.specificity(url) < 0.3])

    x = featurizer.make_features(dataset.qps_trn_ben, use_torch=True)
    pred = model.predict_prob(x, Label.M)

    for user, qp, prob_m in zip(dataset.users_trn_ben, dataset.qps_trn_ben, pred):
        if float(prob_m * 100) < threshold:
            continue

        print("=" * 50)
        print("User: %s" % user)
        print("Probability: %5.2f%%" % float(prob_m * 100))
        print("Scores Histogram:")
        print(histogram.make_features([qp], use_torch=True))
        print("Scores Counts:")
        print(counts.make_features([qp], use_torch=True))

        print("Malicious URLS:")
        pprint([url for url in set(qp.queries) if dataset.specificity(url) < 0.3])


################
#
# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments/http_fee_ctu/fgsm_more_features/"

experiment_filepath: str = False
requests_filepath = "../data/trend_micro_full/user_queries.csv"
scores_filepath = "../data/trend_micro_full/url_scores.csv"

critical_urls_filepath = "../data/trend_micro_full/critical_urls.csv"
# experiment_filepath = "../../results/experiments/trend_micro_full/langrange_net_fgsm_FPR_0.1_adjusted_grad_new_features_l_u_0.05_lr_alpha_0.1_wtf/"

# requests_filepath = "data/user_queries.csv"
# scores_filepath = "data/url_scores.csv"
# critical_urls_filepath = "data/critical_urls.csv"

##########

experiment_filepath = "../../results/experiments/trend_micro_full/knn_fgsm_FPR_0.1"

dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)
histogram = Histogram(dataset.specificity, 3)
counts = BinCounter(dataset.specificity, 3)
#################


model, featurizer, training_attacker = ExperimentHelper.load(experiment_filepath)
print_outliers(10)
print("*"*100)