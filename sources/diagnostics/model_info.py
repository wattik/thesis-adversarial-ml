import os
from pprint import pprint
from random import seed

# from features.features_creator import FeaturesComposer, SpecificityHistogram, EntriesCount
from show import ExperimentHelper

seed(42)

###########
experiment_filepath: str = False
################
#
# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments_config/http_fee_ctu/fgsm_more_features/"

requests_filepath = "data/trend_micro_full/user_queries.csv"
scores_filepath = "data/trend_micro_full/url_scores.csv"
critical_urls_filepath = "data/trend_micro_full/critical_urls.csv"
experiment_filepath = "../../results/experiments_config/trend_micro_full/" \
                      "langrange_net_fgsm_FPR_0.01_cont_2"

# requests_filepath = "data/user_queries.csv"
# scores_filepath = "data/url_scores.csv"
# critical_urls_filepath = "data/critical_urls.csv"

##########

# dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)


#################
model, featurizer, attacker = ExperimentHelper.load(experiment_filepath)

print("MODEL:")
pprint(vars(model))

if hasattr(model, "trainer"):
    print("TRAINER:")
    pprint(vars(model.trainer))

if attacker:
    print("ATTACKER:")
    pprint(vars(attacker))

print("FEATURIZER")
pprint(vars(featurizer))


with open(os.path.join(experiment_filepath, "std.out.txt")) as file:
    lines = list(file)
    print("".join(lines[-40:-1]))
