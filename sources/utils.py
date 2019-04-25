import numpy as np


def to_one_hot(a, n_class=2):
    y = np.zeros((len(a), n_class), dtype=np.int)
    y[np.arange(len(a)), a] = 1

    return y


def from_one_hot(a):
    return np.argmax(a, axis=1)


def reorder(l, idx):
    c = [None] * len(l)
    for i, j in enumerate(idx):
        c[i] = l[j]
    return c


if __name__ == '__main__':
    a = np.array([0, 1, 0, 1])
    print(from_one_hot(to_one_hot(a)))
