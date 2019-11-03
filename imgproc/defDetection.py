"""
Created by JR

This is a demo for processing the images.
we write three classes

1. defDetection involves the basic methods
2. FrontEndPlaneDefectDetection involves specific methods
3. RuningTimeLoop works as main function

The logic is transforming the image into polar coordinate
and processing the image accordingly.

"""

import cv2
import numpy as np
import pickle
from imutils import paths
import matplotlib.pyplot as plt


# this is a base class for defect detection
class defDetection(object):

    def binarizeImage(self, src, thresh):
        """ binarize the image """
        _, thresh = cv2.threshold(src, thresh, 255, cv2.THRESH_BINARY)
        return thresh

    def getImageShape(self, src):
        """ get the shape of the image [height, width] """
        return src.shape[:2]

    def getOuterEdegPoints(self, src):
        """ get the edge pints by scanning the edge """

        pts = []
        h, w = self.getImageShape(src)
        for i in range(w):
            for j in range(h):
                if src[j, i] > 200:
                    pts.append([i, j])
                    break
        return pts

    def getInnerEdgePoints(self, src):
        """ get the edge points by scanning the edge """

        pts = []
        h, w = self.getImageShape(src)
        mid = w // 2
        for i in range(h-1, 0, -1):
            if src[i, mid] < 20:
                for j in range(mid, 0, -1):
                    if src[i, j] > 200:
                        pts.append([j, i])
                        break

                for k in range(mid, w, 1):
                    if src[i, k] > 200:
                        pts.append([k, i])
                        break
            else:
                break
        return pts

    def getCenterandRadius(self, pts):
        """ calculate the center and radius using least square fitting """

        sum_x = 0.0
        sum_y = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        sum_x3 = 0.0
        sum_y3 = 0.0
        sum_xy = 0.0
        sum_x1y2 = 0.0
        sum_x2y1 = 0.0
        for pt in pts:
            x = pt[0]
            y = pt[1]
            x2 = x * x
            y2 = y * y
            sum_x += x
            sum_y += y
            sum_x2 += x2
            sum_y2 += y2
            sum_x3 += x2 * x
            sum_y3 += y2 * y
            sum_xy += x * y
            sum_x1y2 += x * y2
            sum_x2y1 += x2 * y

        N = len(pts)
        C = N * sum_x2 - sum_x * sum_x
        D = N * sum_xy - sum_x * sum_y
        E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x
        G = N * sum_y2 - sum_y * sum_y
        H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y
        a = (H * D - E * G) / (C * G - D * D)
        b = (H * C - E * D) / (D * D - G * C)
        c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N

        center_x = a / (-2)
        center_y = b / (-2)
        radius = np.sqrt(a * a + b * b - 4 * c) / 2

        return [center_x, center_y], radius

    def cart2polar(self, src, center, rMin, rMax, thetaMin, thetaMax):
        """ transform x-y into rho-theta """

        h = int(np.ceil(rMax - rMin))
        w = int(np.ceil(rMax * (thetaMax - thetaMin)))
        dst = np.zeros((h, w), np.uint8)

        delta_r = 1
        delta_theta = (thetaMax - thetaMin) / w
        center_x = center[0]
        center_y = center[1]

        for i in range(w):

            theta = thetaMin + i * delta_theta
            sin_theta = np.sin(theta + np.pi)
            cos_theta = np.cos(theta + np.pi)

            for j in range(h):
                r = int(rMin) + j * delta_r
                x = int(center_x + r * cos_theta)
                y = int(center_y + r * sin_theta)

                if -1 < x < src.shape[1] and -1 < y < src.shape[0]:
                    dst[j, i] = src[y, x]
        return dst

    def polar2cart(self, point, center, radius, thetaOffset):
        """ transform rho-theta to x-y """

        if thetaOffset > 0:
            x = center[0] + point[1] * np.cos(point[0] / radius + thetaOffset + np.pi)
            y = center[1] + point[1] * np.sin(point[0] / radius + thetaOffset + np.pi)
        else:
            x = center[0] + point[1] * np.cos(point[0] / radius)
            y = center[1] + point[1] * np.sin(point[0] / radius)

        return [y, x]

    def transform(self, src):

        thresh = self.binarizeImage(src, 100)
        outer_center, outer_radius = self.getOuterEdegPoints(thresh)
        inner_center, inner_radius = self.getInnerEdgePoints(thresh)
        dst = self.cart2polar(src, outer_center, inner_radius, outer_radius, 0.96, 2.18)

        inner_roi = self.getROI(dst, inner_radius, 1.2*inner_radius)
        outer_roi = self.getROI(dst, 0.9 * outer_radius, outer_radius)
        rivet_roi = self.getROI(dst, 1.2*inner_radius, 0.9 * outer_radius)
        return inner_roi, rivet_roi, outer_roi

    def getROI(self, src, lower_bound, upper_bound):
        """ do not exceed the index of the matrix """
        return src[lower_bound:upper_bound]


# this is the derived class
class FrontEndPlaneDefectDetection(defDetection):

    def loadImage(self, path):
        print("[INF0] loading image paths ... ")
        image_path = list(paths.list_images(path))

        if image_path is None:
            print("[INFO] image paths are empty! Please check the image paths.")
        else:
            data = self.load(image_path)
            print("[INFO] storing data in {}".format("data/data.pkl"))
            with open("dataset/data.pkl", "wb") as fout:
                pickle.dump(data, fout)

    def load(self, img_paths):
        data = []

        for (i, p) in enumerate(img_paths):
            print("[INF0] processed {}/{}".format(i + 1, len(img_paths)))
            image = cv2.imread(p, 0)
            if image is None:
                print("[INF0]", i, "th image is loaded unsuccessfully ... ")
            else:
                print("[INF0]", i, "th image is loaded successfully ... ")
                data.append(image)
        return np.array(data)

    def loadData(self, path):

        print("[INFO] loading data ... ")
        with open(path, "rb") as fin:
            images = pickle.load(fin)
        return images

    def defectDetection(self, src):
        inner_roi, rivet_roi, outer_roi = self.transform(src)


# this is the running time
class RunningTimeLoop(object):

    def __init__(self, path):
        self._path = path
        self._defDetection = defDetection()
        self._fepdd = FrontEndPlaneDefectDetection()

    def run(self):

        data = self._fepdd.loadData(self._path)
        print("[INFO] Loading data finished ... ")

        for img_original in data:
            print("[INFO] cropping image ... ")
            img_original = img_original[20:1060, 20:1900]

            mean = np.mean(img_original)
            print("[INFO] the average of the image is {}".format(mean))

            if mean > 40:
                print("[INFO] binarizing image ... ")
                img_binary = self._defDetection.binarizeImage(img_original, mean)

                pts = self._defDetection.getOuterEdegPoints(img_binary)
                outer_center, outer_radius = self._defDetection.getCenterandRadius(pts)
                print("[INFO] getting center = {}, radius = {}".format(outer_center, outer_radius))

                pts = self._defDetection.getInnerEdgePoints(img_binary)
                inner_center, inner_radius = self._defDetection.getCenterandRadius(pts)
                print("[INFO] getting center = {}, radius = {}".format(inner_center, inner_radius))

                print("[INFO] transforming image ... ")
                img_transformed = self._defDetection.cart2polar(img_original, outer_center,
                                                                inner_radius - 10, outer_radius + 10,
                                                                0.96, 2.18)

                inner_roi = self._defDetection.getROI(img_transformed, 0, int(0.24 * inner_radius) + 10)
                outer_roi = self._defDetection.getROI(img_transformed, int(0.87 * outer_radius - inner_radius + 10),
                                                      int(outer_radius + 10 - inner_radius + 10))
                rivet_roi = self._defDetection.getROI(img_transformed, int(0.2 * inner_radius + 10),
                                                      int(0.9 * outer_radius - inner_radius + 10))

                mean = np.mean(inner_roi)
                inner_roi_bin = self._defDetection.binarizeImage(inner_roi, 80)

                mean = np.mean(rivet_roi)
                rivet_roi_bin = self._defDetection.binarizeImage(rivet_roi, 50)

                mean = np.mean(outer_roi)
                outer_roi = outer_roi - mean * np.ones(outer_roi.shape)
                # outer_roi_bin = self._defDetection.binarizeImage(outer_roi, 0)
                # outer_roi = cv2.blur(outer_roi, (5, 5))
                # outer_roi_bin = cv2.adaptiveThreshold(outer_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                #                                       cv2.THRESH_BINARY, 11, mean * 0.1)
                outer_roi_bin = cv2.Laplacian(outer_roi, cv2.CV_64F)
                outer_roi_binx = cv2.Sobel(outer_roi, cv2.CV_64F, 1, 0, ksize=5)
                outer_roi_biny = cv2.Sobel(outer_roi, cv2.CV_64F, 0, 1, ksize=5)

                plt.figure(1)
                plt.imshow(outer_roi, cmap='gray')

                # plt.figure(2)
                # plt.subplot(3, 2, 1)
                # plt.imshow(inner_roi, cmap='gray')
                # # plt.title("inner_roi")
                #
                # plt.subplot(3, 2, 3)
                # plt.imshow(rivet_roi, cmap='gray')
                # # plt.title("rivet_roi")
                #
                # plt.subplot(3, 2, 5)
                # plt.imshow(outer_roi, cmap='gray')
                # # plt.title("outer_roi")
                #
                # plt.subplot(3, 2, 2)
                # plt.imshow(inner_roi_bin, cmap='gray')
                # # plt.title("inner_roi")
                #
                # plt.subplot(3, 2, 4)
                # plt.imshow(rivet_roi_bin, cmap='gray')
                # # plt.title("rivet_roi")
                #
                # plt.subplot(3, 2, 6)
                # plt.imshow(outer_roi_bin, cmap='gray')
                # # plt.title("outer_roi")

                # corr = outer_roi_bin.dot(outer_roi_bin.T)
                plt.figure(3)
                plt.imshow(outer_roi_bin, cmap='gray')

                plt.draw()
                plt.pause(0.01)
                # print("[INFO] displaying image ... ")
                # item = cv2.resize(img_transformed, None, None, fx=0.5, fy=0.5)
                # cv2.imshow("image", item)
                # cv2.waitKey(0)
                # plt.show()


if __name__ == "__main__":
    RunningTimeLoop("dataset/data.pkl").run()
