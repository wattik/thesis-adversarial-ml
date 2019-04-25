class Request:
    def __init__(
            self,
            time: int,
            query: str
    ):
        self.query = query
        self.time = time


class QueryProfile:
    def __init__(self):
        self.stream = []

    def add(self, request: Request):
        self.stream.append(request)
        return self

    @property
    def queries(self):
        return [request.query for request in self.stream]

    def copy(self):
        qp = type(self)()
        qp.stream = self.stream[:]
        return qp


class AdversarialQueryProfile(QueryProfile):
    def __init__(self, base_qp: QueryProfile):
        super().__init__()
        self.base_qp = base_qp

    @property
    def queries(self):
        return self.base_qp.queries + [request.query for request in self.stream]

    def copy(self):
        qp = type(self)(self.base_qp)
        qp.stream = self.stream[:]
        return qp


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
