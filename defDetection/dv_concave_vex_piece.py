from dv_basic_functions import BasicLib, dv_left, dv_right, dv_y
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ConcaveVexPiece(BasicLib):

    def concaveVexPieceDetection(self, src):

        self.showImage("src", src)
        h, w = self.getImageShape(src)
        binary_thresh = 70
        area_thresh = 2000
        binary_left_roi = self.binariezImage(src, binary_thresh, cv2.THRESH_BINARY)
        binary_left_roi = self.removeNoise(binary_left_roi, area_thresh)
        left_points = self.getEdgePoints(binary_left_roi, h//6, 5*h//6, 200, dv_left)
        binary_thresh = 240
        binary_right_roi = self.binariezImage(src, binary_thresh, cv2.THRESH_BINARY)
        binary_right_roi = self.removeNoise(binary_right_roi, area_thresh)
        right_points = self.getEdgePoints(binary_right_roi, h // 6, 5 * h // 6, 200, dv_right)
        points = np.vstack((left_points, right_points))

        result_image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        for pt in points:
            cv2.circle(result_image, (np.int32(pt[0]), np.int32(pt[1])), 0, (0, 255, 0), 2)

        center, axes, phi = self.fitEllipse(points)
        rotated_image = self.rotateEllipse(src, center, phi)
        self.showImage("rotated_image", rotated_image)

        stretched_image = self.stretchEllipse(rotated_image, center, axes[0] / axes[1])
        self.showImage("stretched_image", stretched_image)

        # cv2.ellipse(result_image, tuple((np.int32(center[0]), np.int32(center[1]))),
        #             tuple((np.int32(axes[0]), np.int32(axes[1]))), phi/np.pi*180, 0, 360, (0, 255, 255), 2)
        # self.showImage("resulting_image", result_image)

        transformed_image = self.cart2polar(stretched_image, center, 100, 500, 3.14, 6.28)
        self.showImage("transformed_image", transformed_image)

        projection_y = self.getProjection(transformed_image, dv_y)
        projection_thresh = 150
        p_area_thresh = 5
        boundaries = self.getBoundaryValues(transformed_image, projection_thresh, p_area_thresh)
        print("[INFO] boundaries = {}".format(boundaries))
        self.showImage("roi", self.getROI(transformed_image, boundaries[1], boundaries[2]))


    def fitEllipse(self, points):

        X = np.zeros((len(points), 5))
        b = np.zeros((5, 1))

        x = points[:, 0]
        y = points[:, 1]

        X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4] = x * y, y * y, x, y, 1
        b = X.transpose().dot(x * x)
        param = np.linalg.inv(X.transpose().dot(X)).dot(-b)

        A = param[0]
        B = param[1]
        C = param[2]
        D = param[3]
        E = param[4]

        delta = A * A- 4 * B
        x0 = (2 * B * C - A * D) / delta
        y0 = (2 * D - A * C) / delta

        num = 2 * (A * C * D - B * C * C - D * D + 4 * B * E - A * A * E)
        den = np.sqrt(A * A + (1 - B) * (1 - B))

        a = np.sqrt(num / (delta * (B + 1 - den)))
        b = np.sqrt(num / (delta * (B + 1 + den)))

        phi = np.arctan(np.sqrt((a * a - b * b * B) / (a * a * B - b * b)))
        return (x0, y0), (a, b), phi
    
    def rotateEllipse(self, src, center, phi):
        h, w = self.getImageShape(src)
        rotated_matrix = cv2.getRotationMatrix2D((center[0], center[1]), -phi*180/np.pi, 1)
        dst = cv2.warpAffine(src, rotated_matrix, (w, h))
        return dst

    def scaleImage(self, src, scaling_coeff):
        h, w = self.getImageShape(src)
        dst = cv2.resize(src, None, fx=1, fy=scaling_coeff, interpolation=cv2.INTER_LINEAR)
        return dst

    def stretchEllipse(self, src, center, scaling_coeff):
        h, w = self.getImageShape(src)
        y0 = np.int32(center[1])
        dst = 0 * src
        for i in range(h):
            temp = np.int32((i - y0) * scaling_coeff + y0)
            temp1 = np.int32((i + 1 - y0) * scaling_coeff + y0)

            if temp < 0:
                temp = 0
            if temp >= h:
                temp = h - 1
            if temp1 < 0:
                temp1 = 0
            if temp1 >= h:
                temp1 = h - 1

            if temp1 - temp > 1:
                dst[temp, :] = src[i, :]
                j = temp + 1
                while j < temp1:
                    dst[j, :] = src[i, :]
                    j += 1
            else:
                dst[temp, :] = src[i, :]
        return dst

