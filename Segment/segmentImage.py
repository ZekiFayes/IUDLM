import numpy as np
import cv2
from segmentGraph import SegmentGraph
import matplotlib.pyplot as plt


class SegmentImage(object):

    def __init__(self):
        self._width = None
        self._height = None
        self._sigma = 0.8
        self._edges = []
        self._num = 0

    def get_size_of_image(self, src):
        dim = src.shape
        self._width = dim[1]
        self._height = dim[0]

    def smooth(self, src):
        return cv2.GaussianBlur(src, ksize=(5, 5), sigmaX=self._sigma)

    def diff(self, src, x1, y1, x2, y2):
        dist = np.sqrt(np.sum(np.square(src[x1, y1] - src[x2, y2])))
        return dist

    def get_edges(self, src):

        for y in range(self._height):
            for x in range(self._width):

                if x < self._width - 1:
                    a = y * self._width + x
                    b = y * self._width + (x + 1)
                    w = self.diff(src, y, x, y, x + 1)
                    self._edges.append([w, a, b])
                    self._num += 1

                if y < self._height - 1:
                    a = y * self._width + x
                    b = (y + 1) * self._width + x
                    w = self.diff(src, y, x, y + 1, x)
                    self._edges.append([w, a, b])
                    self._num += 1

                if x < self._width - 1 and y < self._height - 1:
                    a = y * self._width + x
                    b = (y + 1) * self._width + (x + 1)
                    w = self.diff(src, y, x, y + 1, x + 1)
                    self._edges.append([w, a, b])
                    self._num += 1

                if x < self._width - 1 and y > 0:
                    a = y * self._width + x
                    b = (y - 1) * self._width + (x + 1)
                    w = self.diff(src, y, x, y - 1, x + 1)
                    self._edges.append([w, a, b])
                    self._num += 1

    def segmentImage(self, src):

        plt.figure(1)
        plt.imshow(src)
        self.get_size_of_image(src)
        smooth_src = self.smooth(src)
        self.get_edges(smooth_src)
        self._segmentGraph = SegmentGraph(src, self._width * self._height, self._num, 200, 700)
        self._segmentGraph.segmentGraph(self._edges)
