import pickle
from random import seed

from dataset import Dataset
from evaluate import accuracy
from models.sampling.sampler import TensorSampler, Sampler
from show import ExperimentHelper

seed(42)

###########
experiment_filepath: str = False
################

# requests_filepath = "data/http_fee_ctu/user_queries.csv"
# scores_filepath = "data/http_fee_ctu/url_scores.csv"
# critical_urls_filepath = "data/http_fee_ctu/critical_urls.csv"
# experiment_filepath = "../results/experiments_config/http_fee_ctu/test/"

requests_filepath = "../data/trend_micro_full/user_queries.csv"
scores_filepath = "../data/trend_micro_full/url_scores.csv"
critical_urls_filepath = "../data/trend_micro_full/critical_urls.csv"

experiment_filepath = "../../results/experiments_config/trend_micro_full/langrange_net_fgsm_FPR_0.01_cont_2/"

##########
source_path = "../../results/experiments_config/trend_micro_full/langrange_net_fgsm_FPR_0.01_cont_2/"

model, featurizer, attacker = ExperimentHelper.load(source_path)
dataset = Dataset(scores_filepath, requests_filepath, critical_urls_filepath)

model.trainer.learning_rate = 0.001
# model.trainer.batch_size = 32
# model.trainer.fp_thresh = 0.01/100
# model.trainer.lambda_learning_rate = 5.0
model.model.lam = 500.0
model.trainer.attacker.max_attack_iters = 800

# #######################################

benign_samples = TensorSampler([featurizer.make_features(qp, use_torch=True) for qp in dataset.qps_trn_ben])
malicious_samples = Sampler(dataset.qps_trn_mal)

helper = ExperimentHelper(
    featurizer,
    dataset.qps_trn,
    save_folder=experiment_filepath
)

lambdas = pickle.load(open(source_path + "lambdas.pickle", "rb"))
att_results_tst = pickle.load(open(source_path + "att_res_tst.pickle", "rb"))
benign_accuracy_trn = pickle.load(open(source_path + "benign_accuracy_trn.pickle", "rb"))
benign_accuracy_tst = pickle.load(open(source_path + "benign_accuracy_tst.pickle", "rb"))

for i in range(len(lambdas), 500):
    helper.log("=================================================")
    helper.log("Epoch: %d" % i)

    helper.log("Fitting.")
    model.fit_samplers(benign_samples, malicious_samples)

    helper.log("TRN RESULTS.")
    accuracy_trn = accuracy(model, dataset.qps_trn_ben, dataset.labels_trn_ben)
    lam = float(model.model.lam)

    helper.log("Accuracy on Benign Data: %5.2f %%" % (accuracy_trn * 100))
    helper.log("Lambda: %5.2f" % lam)
    helper.log("p(B): %7.4f" % (lam / (1 + lam)))
    helper.log()
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
    att_results_tst.append(att_res_tst)
    benign_accuracy_trn.append(accuracy_trn)
    benign_accuracy_tst.append(accuracy_tst)

    helper.save_data({
        "lambdas": lambdas,
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
