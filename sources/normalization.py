import torch
from torch import Tensor


class Normalizer:
    def __init__(self, n):
        self.mean = torch.zeros(1, n)
        self.std = torch.ones(1, n)

    def fit(self, x: Tensor):
        self.mean = torch.mean(x, dim=0, keepdim=True)
        self.std = torch.std(x, dim=0, keepdim=True)

        self.std[(self.std ** 2) < 1e-6] = 1.0

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std



