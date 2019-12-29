import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

dv_top = 1
dv_bottom = 2
dv_left = 3
dv_right = 4
dv_x = 0
dv_y = 1
dv_line = 1
dv_quadratic = 2
dv_cubic = 3

class BasicLib(object):

    def getImageShape(self, src):
        """
        This function gets the shape of the image.
        We focus on the width and height.
        :param src: the input image
        :return: shape of the image
        """
        return src.shape[:2]

    def binariezImage(self, src, thresh, mode):
        """
        This function binarizes the image to get binaried image
        :param src: the input image
        :param thresh: thresh for binarizing image
        :param mode: BINARY
        :return: binary image
        """
        _, binary_image = cv2.threshold(src, thresh, 255, mode)
        return binary_image

    def removeNoise(self, src, area_thresh):
        """
        This function removes small area
        :param src: the input image
        :param area_thresh: threshold value for removing small connected components
        :return: the output image
        """
        colors = []
        h, w = self.getImageShape(src)
        retval, labels, stats, centroid = cv2.connectedComponentsWithStats(src)
        colors.append(0)
        for i in range(1, retval):
            if stats[i, cv2.CC_STAT_AREA] < area_thresh:
                colors.append(0)
            else:
                colors.append(255)

        for i in range(h):
            for j in range(w):
                label = labels[i, j]
                src[i, j] = colors[label]

        dst = src
        return dst

    def getContours(self, src, area_thresh):
        """
        This function gets contours of the image
        :param src:  the input image
        :param area_thresh: the threshold value for area
        :return: the output contours
        """
        contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > area_thresh:
                new_contours.append(cnt)

        return new_contours

    def getMaxBoundingRectOfContoursInImage(self, src):
        """
        This function gets the maximum bounding rect of contours in a binarize image
        :param src: the input image
        :return: bounding rect
        """
        contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rect = (0, 0, 0, 0)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if rect[2] < w and rect[3] < h:
                rect = (x, y, w, h)
        return rect

    def getEdgePoints(self, src, starting_point, end_point, thresh, mode):

        """
        this function get the left points of the object in the binarized image
        :param src: the input image
        :param starting_point: starting row
        :param end_point: end row
        :param thresh: threshold value
        :param mode: for 4 directions. top, bottom, left, right
        :return: array
        """
        points = []
        h, w = self.getImageShape(src)

        if mode is dv_left:
            for i in range(starting_point, end_point+1, 1):
                for j in range(w):
                    if src[i, j] >= thresh:
                        points.append([j, i])
                        break

        elif mode is dv_right:
            for i in range(starting_point, end_point+1, 1):
                for j in range(w-1, -1, -1):
                    if src[i, j] >= thresh:
                        points.append([j, i])
                        break

        elif mode is dv_top:
            for j in range(starting_point, end_point+1, 1):
                for i in range(h):
                    if src[i, j] >= thresh:
                        points.append([j, i])
                        break

        elif mode is dv_bottom:
            for j in range(starting_point, end_point+1, 1):
                for i in range(h-1, -1, -1):
                    if src[i, j] >= thresh:
                        points.append([j, i])
                        break

        return np.array(points)

    def getCenter(self, src):
        """
        this function gets the center of the circle using line scanning
        :param src: the input image
        :return: center
        """
        h, w = self.getImageShape(src)
        top_points = self.getEdgePoints(src, w//4, 3*w//4, 200, dv_top)
        bottom_points = self.getEdgePoints(src, w//4, 3*w//4, 200, dv_bottom)
        left_points = self.getEdgePoints(src, h//4, 3*h//4, 200, dv_left)
        right_points = self.getEdgePoints(src, h//4, 3*h//4, 200, dv_right)

        x = (np.mean(left_points, 0)[0] + np.mean(right_points, 0)[0]) / 2
        y = (np.mean(top_points, 0)[1] + np.mean(bottom_points, 0)[1]) / 2

        return [x, y]

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
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(h):
                r = int(rMin) + j * delta_r
                x = int(center_x + r * cos_theta)
                y = int(center_y + r * sin_theta)

                if -1 < x < src.shape[1] and -1 < y < src.shape[0]:
                    dst[j, i] = src[y, x]
        return dst

    def getProjection(self, src, mode):
        """
        this function gets the projection
        :param src:
        :param mode:
        :return:
        """
        if mode is dv_x:
            return np.mean(src, 0)
        elif mode is dv_y:
            return np.mean(src, 1)

    def binarizeProjectionAndNoiseRemoval(self, projection, proj_thresh, width_thresh):
        i = 0
        while i < len(projection):
            if projection[i] >= proj_thresh:
                j = i + 1
                while projection[j] >= proj_thresh:
                    j += 1
                if j - i + 1 < width_thresh:
                    projection[i:j+1] = 0
                i = j + 1
            else:
                projection[i] = 0
                i += 1
        return projection

    def getBoundaries(self, projection, thresh):

        round_flag = False
        boundaries = []
        for i, p in enumerate(projection):

            if not round_flag:
                if p >= thresh:
                    boundaries.append(i)
                    round_flag = True
            else:
                if p <= thresh:
                    boundaries.append(i)
                    round_flag = False

        return boundaries

    def getBoundary(self, projection, thresh, mode):

        if mode is dv_left:
            for i, p in enumerate(projection):
                if p >= thresh:
                    return i
        elif mode is dv_right:
            for i in range(len(projection)-1, -1, -1):
                if projection[i] >= thresh:
                    return i

    def polar2cart(self, point, center, radius, thetaOffset):
        """ transform rho-theta to x-y """

        if thetaOffset > 0:
            x = center[0] + point[1] * np.cos(point[0] / radius + thetaOffset + np.pi)
            y = center[1] + point[1] * np.sin(point[0] / radius + thetaOffset + np.pi)
        else:
            x = center[0] + point[1] * np.cos(point[0] / radius)
            y = center[1] + point[1] * np.sin(point[0] / radius)

        return [y, x]

    def getROI(self, src, lower_bound, upper_bound):
        return src[lower_bound:upper_bound]

    def getROIWithRect(self, src, rect):
        x, y, w, h = rect
        return src[y:y+h, x:x+w]

    def swapXY(self, points):
        new_points = np.zeros(points.shape)
        for i, [x, y] in enumerate(points):
            new_points[i, :] = [y, x]
        return new_points

    def translatePoints(self, points, dist):
        new_points = np.zeros(points.shape)
        for i, [x, y] in enumerate(points):
            new_points[i, :] = [x + dist, y]
        return new_points

    def polynomialFitting(self, points, mode):

        if mode == dv_quadratic:

            X = np.zeros((len(points), 3))
            b = np.zeros((3, 1))
            x = points[:, 0]
            y = points[:, 1]
            X[:, 0], X[:, 1], X[:, 2] = 1, x, x * x
            b = X.transpose().dot(y)
            param = np.linalg.inv(X.transpose().dot(X)).dot(b)

            a = param[2]
            b = param[1]
            c = param[0]

            points_hat = np.zeros(points.shape)
            for i in range(len(points)):
                points_hat[i, 0] = points[i, 0]
                points_hat[i, 1] = a * points[i, 0] * points[i, 0] + b * points[i, 0] + c
            return points_hat

    def scanRows(self, src, left_points, right_points, delta):
        dst = np.zeros(src.shape)
        for i in range(len(left_points)):
            row = np.int32(left_points[i, 1])
            col1 = np.int32(left_points[i, 0]+1)
            col2 = np.int32(right_points[i, 0] + 1)
            temp = np.mean(src[row, col1:col2])
            dst[row, col1:col2] = src[row, col1:col2] < (temp - delta)
        return dst

    def scanCols(self, src, top_points, bottom_points, delta):
        dst = np.zeros(src.shape)
        for i in range(len(top_points)):
            col = np.int32(top_points[i, 0])
            row1 = np.int32(top_points[i, 1]+1)
            row2 = np.int32(bottom_points[i, 1] + 1)
            temp = np.mean(src[row1:row2, col])
            dst[row1:row2, col] = src[row1:row2, col] < (temp - delta)
        return dst

    def showImage(self, name, image):
        cv2.imshow(name, image)
        cv2.waitKey(0)

    def showContours(self, src, contours):

        image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            cv2.rectangle(image, rect, (0, 255, 0), 1)
        self.showImage("image", image)

    def show3DImage(self, src, stride):

        fig = plt.figure()
        ax = Axes3D(fig)
        x = np.arange(0, src.shape[1], 1)
        y = np.arange(0, src.shape[0], 1)
        x, y = np.meshgrid(x, y)
        z = src
        ax.plot_surface(x, y, z, rstride=stride, cstride=stride, cmap='rainbow')
        plt.show()

    def getHist(self, src):

        hist = cv2.calcHist([src], [0], None, [256], [0, 256])
        # hist = hist[0:250]
        normalized_hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX)
        return normalized_hist

    def getWidthOfHistogram(self, hist, threshold):

        """ this is to find the width when give the threshold """
        hist_length = 250
        start = 0
        end = hist_length
        start_stop = False
        end_stop = False
        # width = 0
        while (start < end) and (not start_stop or not end_stop):
            if not start_stop:
                if hist[start] > threshold:
                    start_stop = True
                else:
                    start += 1

            if not end_stop:
                if hist[end] > threshold:
                    end_stop = True
                else:
                    end -= 1

        if start >= end:
            width = 0
            mu = 127
        else:
            width = end - start
            mu = (start + end) / 2

        return width, mu

    def autoThreshforDarkSpotDetection(self, src, std_factor, std_kernel_size, mean_factor, mean_kernel_size, bias):

        roi = cv2.blur(src, (3, 3))
        roi = np.float32(roi)
        roi /= 255.0
        # dst = np.zeros(roi.shape)
        std_m = np.zeros(roi.shape)
        mean_m = np.zeros(roi.shape)

        if std_factor > 0:

            std_kernel = np.ones(std_kernel_size, np.float32)
            n = np.sum(np.sum(std_kernel))

            temp_mean = cv2.filter2D(roi, -1, std_kernel)
            temp_mean = np.multiply(temp_mean, temp_mean) / (n * n)
            variance_m = cv2.filter2D(np.multiply(roi, roi), -1, std_kernel) / n
            temp = np.where(variance_m - temp_mean > np.zeros(roi.shape), variance_m - temp_mean, np.zeros(roi.shape))
            std_m = np.sqrt(temp) * 255.0

        if mean_factor > 0:
            mean_kernel = np.ones(mean_kernel_size, np.float32)
            mean_m = cv2.filter2D(roi, -1, mean_kernel)

        if std_factor == 0 and mean_factor == 0:
            if bias == 0:
                bias = 70
            dst = bias
        else:
            dst = std_m * std_factor + mean_m * mean_factor + bias

        return dst

    def MIN(self, arr, thresh):
        thresh = np.ones(arr.shape) * thresh
        return np.where(arr < thresh, arr, thresh)

    def MAX(self, arr, thresh):
        thresh = np.ones(arr.shape) * thresh
        return np.where(arr > thresh, arr, thresh)

    def binarizeArray(self, arr, thresh_arr, mode):
        ONE = np.ones((arr.shape[0], 1))
        ZERO = np.zeros((arr.shape[0], 1))
        if mode == 'greater_than':
            temp = np.where(arr > thresh_arr, ONE, ZERO)
        elif mode == 'less_than':
            temp = np.where(arr < thresh_arr, ONE, ZERO)
        return temp

    def detectScratching(self, src):
        # src = cv2.blur(src, (5, 5))
        dst = np.zeros(src.shape)
        num_of_cols_as_a_whole = 4
        num_of_traversal = src.shape[1] // num_of_cols_as_a_whole
        print(self.getImageShape(src))

        dark_spot_thresh_lb = 25
        dark_spot_thresh_ub = 70
        dark_spot_coefficient = 0.4

        bright_spot_thresh_lb = 180
        bright_spot_thresh_ub = 230
        bright_spot_coefficient = 2

        for i in range(num_of_traversal):
            temp_ones = np.ones((src.shape[0], num_of_cols_as_a_whole))
            temp_arr = src[:, num_of_cols_as_a_whole*i:num_of_cols_as_a_whole*(i+1)]
            temp_mean = np.mean(temp_arr)

            dark_spot_thresh = self.MIN(self.MAX(temp_mean, dark_spot_thresh_lb), dark_spot_thresh_ub)
            bright_spot_thresh = self.MAX(self.MIN(temp_mean, bright_spot_thresh_ub), bright_spot_thresh_lb)
            dst[:, num_of_cols_as_a_whole*i:num_of_cols_as_a_whole*(i+1)] \
                = cv2.bitwise_or(self.binarizeArray(temp_arr, dark_spot_thresh, 'less_than'),
                                 self.binarizeArray(temp_arr, bright_spot_thresh_ub, 'greater_than'))

        return dst

    def preprocessImage(self, src, binary_thresh, area_thresh, min_r, max_r, min_theta, max_theta):

        binary_image = self.binariezImage(src, binary_thresh, cv2.THRESH_BINARY)
        self.showImage("binary_image", binary_image)
        re_binary_image = self.removeNoise(binary_image, area_thresh)
        self.showImage("re_binary_image", re_binary_image)
        rect = self.getMaxBoundingRectOfContoursInImage(re_binary_image)
        roi = self.getROIWithRect(re_binary_image, rect)
        self.showImage("roi", roi)
        center = self.getCenter(roi)
        roi = self.getROIWithRect(src, rect)
        transformed_image = self.cart2polar(roi, center, min_r, max_r, min_theta, max_theta)
        return transformed_image

    def getBoundaryValues(self, src, projection_thresh, area_thresh):

        projection_y = self.getProjection(src, dv_y)
        plt.plot(projection_y)
        binary_projection = self.binarizeProjectionAndNoiseRemoval(projection_y, projection_thresh, area_thresh)
        plt.plot(binary_projection)
        plt.show()
        boundaries = self.getBoundaries(binary_projection, projection_thresh)
        return boundaries

    def scratchingDetection(self, src):

        projection = self.getProjection(src, dv_y)
        plt.figure()
        plt.plot(projection, label='projection')
        plt.legend()
        plt.show()

        roi_ds = self.detectScratching(src)
        self.showImage("roi_ds", roi_ds)

    def histogramDetection(self, src, thresh):

        hist = self.getHist(src)
        plt.figure()
        plt.plot(hist, label='hist')
        plt.show()

        hist_width, hist_mu = self.getWidthOfHistogram(hist, thresh)
        print("[INFO] hist_width = {}, hist_mu = {}".format(hist_width, hist_mu))

    def autoThreshDetection(self, src):

        binary_roi = np.ones(src.shape)
        binary_thresh = self.autoThreshforDarkSpotDetection(src, 4, (3, 3), 0.6, (3, 3), 100)
        binary_roi = np.where(src < binary_thresh, binary_roi, binary_roi * 0)
        self.showImage("binary_roi", binary_roi)
        self.show3DImage(binary_thresh, 5)
