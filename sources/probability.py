class Probability:
    def __init__(self, scores):
        self.scores = scores

    def __call__(self, url):
        try:
            return self.scores[url]
        except KeyError:
            return 0.5
