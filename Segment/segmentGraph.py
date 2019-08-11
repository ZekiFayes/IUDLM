""" This Class involves Segment Graph"""
from operation import Operations
import numpy as np
import matplotlib.pyplot as plt


class SegmentGraph(object):

    def __init__(self, src, num_vertices, num_edges, c, minSize):
        self._image = src
        self._num_vertices = num_vertices
        self._num_edges = num_edges
        self._c = c
        self._minSize = minSize
        self._operation = Operations(num_vertices)

        """ Initialize threshold for each vertex """
        self._thresh = []
        for i in range(self._num_vertices):
            self._thresh.append(self.threshold_fn(1))

    def threshold_fn(self, size):
        return self._c / size

    def segmentGraph(self, edges):
        """ The edges are nested lists [[w, a, b],..., [w, a, b]] """

        sorted_edges = sorted(edges)

        for i in range(self._num_edges):
            a = self._operation.find(sorted_edges[i][1])
            b = self._operation.find(sorted_edges[i][2])
            if a != b:
                if sorted_edges[i][0] <= self._thresh[a] and sorted_edges[i][0] <= self._thresh[b]:
                    self._operation.join(a, b)
                    a = self._operation.find(a)
                    self._thresh[a] = sorted_edges[i][0] + self.threshold_fn(self._operation.size(a))

        for i in range(self._num_edges):
            a = self._operation.find(sorted_edges[i][1])
            b = self._operation.find(sorted_edges[i][2])

            if a != b and (self._operation.size(a) < self._minSize or self._operation.size(b) < self._minSize):
                self._operation.join(a, b)

        # num = self._operation.num_sets()
        # print(num)

        colors = []
        for i in range(self._num_vertices):
            b = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            r = np.random.randint(0, 256)
            colors.append([b, r, g])

        dim = self._image.shape
        dst = self._image.copy()

        for y in range(dim[0]):
            for x in range(dim[1]):
                temp = self._operation.find(y * dim[1] + x)
                dst[y, x] = colors[temp]

        plt.figure(2)
        plt.imshow(dst)
        plt.show()
