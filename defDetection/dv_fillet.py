from dv_basic_functions import BasicLib
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Fillet(BasicLib):

    def filletDetection(self, src):
        binary_thresh, area_thresh = 80, 2000
        min_r, max_r = 250, 700
        min_theta, max_theta = np.pi, 2 * np.pi
        transformed_image = self.preprocessImage(src, binary_thresh, area_thresh, min_r, max_r, min_theta, max_theta)
        self.showImage("transformed_image", transformed_image)
        projection_thresh = 150
        p_area_thresh = 5
        boundaries = self.getBoundaryValues(transformed_image, projection_thresh, p_area_thresh)
        print("[INFO] boundaries = {}".format(boundaries))
        fillet1_roi, fillet2_roi, fillet3_roi, fillet4_roi = self.segmentROI(transformed_image, boundaries)

        # self.scratchingDetection(fillet1_roi)
        # self.histogramDetection(fillet1_roi, 0.5)
        self.autoThreshDetection(fillet1_roi)

    def segmentROI(self, src, boundaries):

        fillet1_roi = self.getROI(src, boundaries[0], boundaries[1])
        fillet2_roi = self.getROI(src, boundaries[2], boundaries[3])
        fillet3_roi = self.getROI(src, boundaries[4], boundaries[5])
        fillet4_roi = self.getROI(src, boundaries[6], boundaries[7])

        return fillet1_roi, fillet2_roi, fillet3_roi, fillet4_roi
