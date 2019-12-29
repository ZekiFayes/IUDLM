from dv_basic_functions import BasicLib
from dv_basic_functions import dv_y
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class EndPlane(BasicLib):

    def endPlaneDetection(self, src):

        binary_thresh, area_thresh = 120, 1000
        min_r, max_r = 120, 380
        min_theta, max_theta = np.pi, 2 * np.pi
        transformed_image = self.preprocessImage(src, binary_thresh, area_thresh, min_r, max_r, min_theta, max_theta)
        self.showImage("transformed_image", transformed_image)

        projection_thresh = 150
        p_area_thresh = 5
        boundaries = self.getBoundaryValues(transformed_image, projection_thresh, p_area_thresh)
        print("[INFO] boundaries = {}".format(boundaries))
        inner_roi, outer_roi = self.segmentROI(transformed_image, boundaries)

        # self.scratchingDetection(inner_roi)
        # self.scratchingDetection(outer_roi)
        #
        # self.histogramDetection(inner_roi, 0.5)
        # self.histogramDetection(outer_roi, 0.5)

        self.autoThreshDetection(inner_roi)
        self.autoThreshDetection(outer_roi)

    def segmentROI(self, src, boundaries):

        inner_roi = self.getROI(src, boundaries[0], boundaries[1])
        outer_roi = self.getROI(src, boundaries[2], boundaries[3])
        return inner_roi, outer_roi
