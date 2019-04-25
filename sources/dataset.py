import csv
from collections import defaultdict
from random import shuffle, random
from typing import List

from models.base import Label
from models.query_profiles import QueryProfile, Request
from probability import Probability
from utils import reorder


class Dataset:
    def __init__(self, scores_filepath, requests_filepath, critical_urls_filepath, split_ratio=0.8):
        self.specificity = load_scores(scores_filepath)

        self.users_ben, self.qps_ben = load_query_profiles(requests_filepath)
        self.labels_ben = make_labels(self.qps_ben, Label.B)

        self.users_mal, self.critical_queries = load_critical_queries(critical_urls_filepath)
        self.qps_mal = make_dummy_adv_query_profiles(self.critical_queries)
        self.labels_mal = make_labels(self.qps_mal, Label.M)

        # Compounded dataset
        k = 1
        self.qps = self.qps_ben + self.qps_mal * k
        self.labels = self.labels_ben + self.labels_mal * k
        self.users = self.users_ben + self.users_mal * k

        # Sizes
        self.n = len(self.users)
        self.n_ben = len(self.users_ben)
        self.n_mal = self.n - self.n_ben

        idx = list(range(self.n))
        shuffle(idx)

        self.qps = reorder(self.qps, idx)
        self.labels = reorder(self.labels, idx)
        self.users = reorder(self.users, idx)

        # Training testing splits
        self.first_tst_idx = int(self.n * split_ratio)

        self.qps_trn = self.qps[:self.first_tst_idx]
        self.labels_trn = self.labels[:self.first_tst_idx]
        self.users_trn = self.users[:self.first_tst_idx]

        self.qps_tst = self.qps[self.first_tst_idx:]
        self.labels_tst = self.labels[self.first_tst_idx:]
        self.users_tst = self.users[self.first_tst_idx:]

        ##########

        self.qps_trn_ben = [qp for qp, l in zip(self.qps_trn, self.labels_trn) if l == Label.B]
        self.labels_trn_ben = [l for l in self.labels_trn if l == Label.B]

        self.qps_trn_mal = [qp for qp, l in zip(self.qps_trn, self.labels_trn) if l == Label.M]
        self.labels_trn_mal = [l for l in self.labels_trn if l == Label.M]

        self.qps_tst_ben = [qp for qp, l in zip(self.qps_tst, self.labels_tst) if l == Label.B]
        self.labels_tst_ben = [l for l in self.labels_tst if l == Label.B]
        self.users_tst_ben = [user for user, l in zip(self.users_tst, self.labels_tst) if l == Label.B]

        self.qps_tst_mal = [qp for qp, l in zip(self.qps_tst, self.labels_tst) if l == Label.M]

        ############

        self.legitimate_queries = [url for qp in self.qps_trn for url in qp.queries if self.specificity(url) > 0.5]
        self.urls = list(set(url for qp in self.qps_trn for url in qp.queries))


#####################################

def load_scores(filepath: str):
    with open(filepath) as file:
        reader = csv.reader(file, delimiter=' ')

        return Probability({url: float(score) for url, score in reader})


def load_query_profiles(filepath: str):
    with open(filepath) as file:
        user_entries = defaultdict(QueryProfile)

        for time, user, url in csv.reader(file, delimiter=' '):
            user_entries[user].add(Request(time, url))

        return list(user_entries.keys()), list(user_entries.values())


def load_critical_queries(filepath: str):
    with open(filepath) as file:
        critical_queries = defaultdict(list)

        for user, query in csv.reader(file, delimiter=' '):
            critical_queries[user].append(query)

        return list(critical_queries.keys()), list(critical_queries.values())


###########################################

def make_labels(query_profiles: List[QueryProfile], default_label: Label) -> List[Label]:
    return [default_label for _ in query_profiles]


def make_dummy_adv_query_profiles(adv_queries: List[List[str]]) -> List[QueryProfile]:
    def basic_query_profile(urls):
        qp = QueryProfile()
        for url in urls:
            qp.add(Request(0, url))
        return qp

    return [basic_query_profile(urls) for urls in adv_queries]
