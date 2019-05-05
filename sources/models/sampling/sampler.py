from typing import List

import torch


class Sampler:
    def __init__(self, samples: List):
        self.samples = samples
        self.length = len(samples)
        self.last_start = 0

        self.concat = lambda x: x[0] + x[1]

    def get_all(self):
        return self.samples

    def next(self, batch_size: int = 1):
        s = self.last_start
        e = min([self.length, self.last_start + batch_size])

        self.last_start = (e % self.length)

        n = e - s
        samples = self.samples[s:e]

        if n < batch_size:
            return self.concat([samples, self.next(batch_size - n)])
        else:
            return samples


class TensorSampler(Sampler):
    def __init__(self, tensors: List[torch.Tensor]):
        super().__init__(torch.cat(tensors))
        self.concat = torch.cat


if __name__ == '__main__':
    sampler = TensorSampler(list(torch.rand(10,3,4)))

    for _ in range(5):
        print(sampler.next(4).shape)
