""" This class involves some basic operations """


class Operations(object):

    def __init__(self, num_vertice):
        self._num = num_vertice
        self._rank = []
        self._size = []
        self._p = []

        for i in range(self._num):
            self._rank.append(0)
            self._size.append(1)
            self._p.append(i)

    def find(self, x):
        y = x
        while y != self._p[y]:
            y = self._p[y]
        self._p[x] = y
        return y

    def join(self, x, y):
        if self._rank[x] > self._rank[y]:
            self._p[y] = x
            self._size[x] += self._size[y]
        else:
            self._p[x] = y
            self._size[y] += self._size[x]

            if self._rank[x] == self._rank[y]:
                self._rank[y] += 1
        self._num -= 1

    def size(self, x):
        return self._size[x]

    def num_sets(self):
        return self._num
