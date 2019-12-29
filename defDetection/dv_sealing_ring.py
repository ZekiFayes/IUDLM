from dv_basic_functions import BasicLib
import cv2
import matplotlib.pyplot as plt
import numpy as np


class SealingRing(BasicLib):

    def sealingRingDetection(self, src):

        binary_thresh, area_thresh = 120, 2000
        min_r, max_r = 90, 240
        min_theta, max_theta = np.pi, 2 * np.pi
        transformed_image = self.preprocessImage(src, binary_thresh, area_thresh, min_r, max_r, min_theta, max_theta)
        self.showImage("transformed_image", transformed_image)
        projection_thresh = 150
        p_area_thresh = 5
        boundaries = self.getBoundaryValues(transformed_image, projection_thresh, p_area_thresh)
        print("[INFO] boundaries = {}".format(boundaries))
        sealing_ring_roi = self.segmentROI(transformed_image, boundaries)

        # self.scratchingDetection(sealing_ring_roi)
        # self.histogramDetection(sealing_ring_roi, 0.5)
        self.autoThreshDetection(sealing_ring_roi)

    def segmentROI(self, src, boundaries):
        sealing_ring_roi = self.getROI(src, boundaries[2], boundaries[3])
        return sealing_ring_roi
