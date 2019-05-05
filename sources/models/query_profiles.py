from typing import List

from utils import shuffle


class Request:
    def __init__(
            self,
            time: int,
            query: str
    ):
        self.query = query
        self.time = time

    def __str__(self):
        return "%5.2f h : %s" % (self.time, self.query)


class QueryProfile:
    def __init__(self):
        self.stream = []
        self.user = None

    def add(self, request: Request):
        self.stream.append(request)
        return self

    @property
    def queries(self) -> List[str]:
        return [request.query for request in self.stream]

    @property
    def requests(self) -> List[Request]:
        return list(self.stream)

    def copy(self):
        qp = type(self)()
        qp.stream = self.stream[:]
        return qp

    def __str__(self):
        s = []

        for request in sorted(self.requests, key=lambda r: r.time):
            s.append(str(request))

        return "\n".join(s)


class AdversarialQueryProfile(QueryProfile):
    def __init__(self, base_qp: QueryProfile):
        super().__init__()
        self.base_qp = base_qp

    @property
    def queries(self):
        return self.base_qp.queries + [request.query for request in self.stream]

    @property
    def requests(self):
        return shuffle(super().requests + self.base_qp.requests)

    def copy(self):
        qp = type(self)(self.base_qp)
        qp.stream = self.stream[:]
        return qp

    def __str__(self):
        s = []

        for request in sorted(self.base_qp.requests, key=lambda r: r.time):
            s.append(str(request) + " " + "(critical)")

        for request in sorted(super().requests, key=lambda r: r.time):
            s.append(str(request))

        return "\n".join(s)


class NoQueriesProfile(QueryProfile):
    def __init__(self, base_qp: AdversarialQueryProfile):
        super().__init__()

        if not isinstance(base_qp, AdversarialQueryProfile):
            raise ValueError("Only AdvQP classes allowed.")

        self.base_qp = base_qp

    def add(self, request: Request):
        return self.base_qp.add(request)

    @property
    def queries(self):
        return self.base_qp.base_qp.queries

    def __str__(self):
        return "No Attack."
