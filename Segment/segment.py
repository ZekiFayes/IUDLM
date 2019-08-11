from segmentImage import SegmentImage
import cv2
import matplotlib.pyplot as plt


class Segment(object):

    def __init__(self, filename):
        self._image = None
        self._fileName = filename
        self._segmentImage = SegmentImage()

    def readImage(self):
        self._image = cv2.imread(self._fileName)

    def segment(self):
        self.readImage()
        self._segmentImage.segmentImage(self._image)

