from collections import deque


class RotDeque(deque):
    def next(self):
        self.rotate(1)
        return self[0]


if __name__ == '__main__':
    q = RotDeque([1, 2, 3])

    for i in range(10):
        print(q.next())
