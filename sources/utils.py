import random
from time import time

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


def shuffle(l):
    return random.sample(l, len(l))


def stop_time(description="Time: "):
    def dec(f):
        def inner_f(*args):
            start_time = time()
            ret = f(*args)
            end_time = time()
            print(description + "%6.2f s" % (end_time - start_time))
            return ret

        return inner_f

    return dec


if __name__ == '__main__':
    a = np.array([0, 1, 0, 1])
    print(from_one_hot(to_one_hot(a)))


    @stop_time
    def a(x):
        return x


    print(a("str"))
