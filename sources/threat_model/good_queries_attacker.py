import datetime
from random import choice
from typing import List

from models.base import ModelBase
from models.query_profiles import QueryProfile, Request
from threat_model.base import Attacker


class GoodQueriesAttacker(Attacker):
    def __init__(self, legitimate_queries: List[str], iteration):
        super().__init__(iteration)
        self.legitimate_queries = legitimate_queries

    def obfuscate(self, model: ModelBase, query_profile: QueryProfile, only_features=True) -> QueryProfile:
        qp_adv = QueryProfile()

        for critical_query in query_profile.queries:
            qp_adv.add(Request(0, critical_query))

        legitimate_query = choice(self.legitimate_queries)
        now = datetime.datetime.now()
        t = int(now.hour)
        qp_adv.add(Request(t, legitimate_query))

        return qp_adv