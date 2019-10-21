import numpy as np
from imutils import paths
import imutils
import cv2
import pickle
import matplotlib.pyplot as plt
import math


class Image(object):

    def __init__(self):
        self.className = ['1', '4', '8']

    def loadImage(self, name):

        """ load the image """
        return cv2.imread(name)

    def binarizeImage(self, src):

        """ binarize the image """
        _, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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

    def getAngleRange(self, pts, center, radius):

        """ get the angle range of an arc """
        pts_sorted = sorted(pts)
        pt1 = pts_sorted[0]
        pt2 = pts_sorted[-1]

        print("[INF0] left point = ", pt1)
        print("[INF0] right point = ", pt2)

        """ this is to calculate the angle and it depends on the point """
        rightAngle = math.atan2((pt1[1] - center[1]), (pt1[0] - center[0]))
        leftAngle  = math.atan2((pt2[1] - center[1]), (pt2[0] - center[0]))

        return -leftAngle, -rightAngle

    def cart2polar(self, src, center, radius):

        """ transform x-y into rho-theta """
        h = int(np.ceil(radius))
        w = int(np.ceil(h * 2 * np.pi))
        dst = np.zeros((h, w), np.uint8)

        delta_r = 1
        delta_theta = 2.0 * np.pi / w

        center_x = center[0]
        center_y = center[1]

        for i in range(w):
            theta = i * delta_theta
            sin_theta = np.sin(theta + np.pi)
            cos_theta = np.cos(theta + np.pi)

            for j in range(h):
                r = j * delta_r
                x = int(center_x + r * cos_theta)
                y = int(center_y + r * sin_theta)
                if -1 < x < src.shape[1] and -1 < y < src.shape[0]:
                    dst[j, i] = src[y, x]

        return dst

    def cart2polar(self, src, center, rLower, rUpper):

        """ transform x-y into rho-theta """
        h = int(np.ceil(rUpper - rLower))
        w = int(np.ceil(rUpper * 2 * np.pi))
        dst = np.zeros((h, w), np.uint8)

        delta_r = 1
        delta_theta = 2.0 * np.pi / w

        center_x = center[0]
        center_y = center[1]

        for i in range(w):
            theta = i * delta_theta
            sin_theta = np.sin(theta + np.pi)
            cos_theta = np.cos(theta + np.pi)

            for j in range(h):
                r = int(rLower) + j * delta_r
                x = int(center_x + r * cos_theta)
                y = int(center_y + r * sin_theta)
                if -1 < x < src.shape[1] and -1 < y < src.shape[0]:
                    dst[j, i] = src[y, x]

        return dst

    def cart2polar(self, src, center, rMin, rMax, theta1, theta2):

        """ transform x-y into rho-theta """
        h = int(np.ceil(rMax - rMin))
        w = int(np.ceil(rMax * (theta2 - theta1)))
        dst = np.zeros((h, w), np.uint8)

        delta_r = 1
        delta_theta = (theta2 - theta1) / w

        center_x = center[0]
        center_y = center[1]

        for i in range(w):
            theta = theta1 + i * delta_theta
            sin_theta = np.sin(theta + np.pi)
            cos_theta = np.cos(theta + np.pi)

            for j in range(h):
                r = int(rMin) + j * delta_r
                x = int(center_x + r * cos_theta)
                y = int(center_y + r * sin_theta)
                if -1 < x < src.shape[1] and -1 < y < src.shape[0]:
                    dst[j, i] = src[y, x]

        return dst

    def polar2cart(self, point, center, radius):

        """ transform rho-theta to x-y """
        x = center[0] + point[1] * np.cos(point[0] / radius)
        y = center[1] + point[1] * np.sin(point[0] / radius)

        return [y, x]

    def shiftImage(self, src, theta_offset):

        """ shift the image by theta """
        height, width = src.shape[:2]
        t_offset = theta_offset * width // 360
        dst = 0 * src

        for i in range(width):
            for j in range(height):
                if i + t_offset < width:
                    dst[j, i + t_offset] = src[j, i]
                else:
                    dst[j, i - width + t_offset] = src[j, i]
        return dst

    def showImage(self, name, src):

        """ show the image """
        resized = cv2.resize(src, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(name, resized)
        cv2.waitKey(0)

    def plotImage(self, num_figure, name, src):

        plt.figure(num_figure)
        plt.imshow(src, cmap='gray')
        plt.title(name)
        plt.draw()
        plt.pause(0.01)

    def findContours(self, src):

        """ find the contours of an image """
        contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def findBoundingBoxes(self, contours):

        """ find the boxes that fit every contour """
        if contours is None:
            return []

        rect_boxes = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            """ constrain the boxes into a relatively small range """
            if 10 < w < 1000:
                rect_boxes.append((x, y, w, h))

        return rect_boxes

    def removeSmallConnectedArea(self, boxes):

        if boxes is None:
            print("there is no box!")
            return []
        else:
            boxes_new = []
            for (x, y, w, h) in boxes:
                if 20 < w < 190 and 20 < h < 130:
                    boxes_new.append((x, y, w, h))
            return boxes_new

    def plotString(self, src, boxes):

        """ plot the contents involved in the box areas """
        if boxes is None:
            print("there is no box!")
            return

        i = 0
        for (x, y, w, h) in boxes:
            roi = src[y:y+h, x:x+w]
            cv2.imshow(str(i), roi)
            cv2.waitKey(0)
            i += 1

    def boundaryChecking(self, boxes):

        box1, boxn = boxes[0], boxes[-1]

        if box1[0] > 20 and boxn[0] + boxn[3] + 20 < 1950:
            return boxes
        else:
            return []

    def detectString(self, boxes):

        """
        traverse every boxes to located the strings and classify the boxes
        this function needs to be optimized
        """
        if boxes is None:
            print("there is no box!")
            return 0, [], []

        # box1 = []
        # box2 = []

        if not len(boxes):
            print("there is no box!")
            return 0, [], []

        elif len(boxes) == 1:
            if boxes[0][0] > 20 and boxes[0][0] + boxes[0][3] + 20 < 1950:
                return 1, boxes, []
            else:
                return 0, [], []

        elif len(boxes) >= 4:

            boxes = np.array(boxes)
            # mean = np.mean(boxes, 0)
            # print("[INF0] mean = ", mean)
            std = np.std(boxes, 0, ddof=1)
            print("[INF0] std = ", std)

            if len(boxes) == 4:
                if std[0] > 500:
                    # for (x, y, w, h) in boxes:
                    #
                    #     if x < mean[0]:
                    #         box1.append((x, y, w, h))
                    #     else:
                    #         box2.append((x, y, w, h))
                    #
                    # print("[INF0] 2 classes")
                    # return 2, box1, box2

                    return 0, [], []
                else:
                    if self.boundaryChecking(boxes) is not None:
                        return 1, boxes, []
                    else:
                        return 0, [], []

            elif len(boxes) == 10:
                if std[0] > 500:
                    # for (x, y, w, h) in boxes:
                    #
                    #     if x < mean[0]:
                    #         box1.append((x, y, w, h))
                    #     else:
                    #         box2.append((x, y, w, h))
                    #
                    # print("[INF0] 2 classes")
                    # return 2, box1, box2
                    return 0, [], []
                else:
                    if self.boundaryChecking(boxes) is not None:
                        return 1, boxes, []
                    else:
                        return 0, [], []
            else:
                return 0, [], []
        else:
            return 0, [], []

    def processImage(self, src, ith_image):

        img = src[20:1060, 20:1900]

        print("[INF0] binarizing image ... ")
        binarized_images = self.binarizeImage(img)

        print("[INF0] fitting circle ... ")
        pts = self.getOuterEdegPoints(binarized_images)
        center, rMax = self.getCenterandRadius(pts)
        print("[INF0] center = ", center)
        print("[INF0] outer radius = ", rMax)

        print("[INF0] calculating angle range ... ")
        theta_min, theta_max = self.getAngleRange(pts, center, rMax)
        print("[INF0] theta_min = ", theta_min)
        print("[INF0] theta_max = ", theta_max)

        pts = self.getInnerEdgePoints(binarized_images)
        center1, rMin = self.getCenterandRadius(pts)
        print("[INF0] center = ", center1)
        print("[INF0] inner radius = ", rMin)

        print("[INF0] transforming image ... ")
        cart_image = self.cart2polar(img, center, rMin, rMax, theta_min+0.03, theta_max-0.03)
        # self.showImage("cart_image", cart_image)
        print(cart_image.shape)

        mean = np.mean(cart_image)

        print("[INF0] thresholding image ... ")
        _, thresh = cv2.threshold(cart_image, int(mean), 255, cv2.THRESH_BINARY_INV)
        roi = thresh[0:140]
        # self.showImage("roi", roi)
        self.plotImage(1, "original image", roi)
        print("[INF0] finding contours ... ")
        contours = self.findContours(roi)

        if contours is not None:
            print("[INF0] finding boxes ... ")
            boxes = self.findBoundingBoxes(contours)

            print("[INFO] removing small connected components ... ")
            boxes = self.removeSmallConnectedArea(boxes)
            print(boxes)

            print("[INF0] sorting boxes ... ")
            sorted_boxes = sorted(boxes)
            print(sorted_boxes)

            print("[INF0] classifying boxes ... ")
            num_classes, box1, box2 = self.detectString(sorted_boxes)

            if num_classes == 1:
                # self.plotString(roi, box1)
                xywh = np.array(box1)
                mean = np.mean(xywh, 0)
                num = int((85 + (mean[0] + mean[2] / 2) / ((theta_max - theta_min) * rMax) * 180 / np.pi) / 2)

                if ith_image + num > 178:
                    num = ith_image + num - 178
                else:
                    num = num + ith_image

                return num

            elif num_classes == 2:
                # self.plotString(roi, box1)
                # self.plotString(roi, box2)
                return -1
            else:
                return -1

    def run(self):

        with open("data/data.pkl", "rb") as fin:
            data = pickle.load(fin)
        #
        # ith_image = np.random.randint(0, 177)
        # print("[INF0] {}/{} is selected".format(ith_image, len(data)-1))
        #
        # img = data[0]
        # self.showImage("image", img)
        #
        # num = self.processImage(img, 0)
        # if num is not -1:
        #     print("[INF0] pick ", num, "th image ")
        #     img = data[num]
        #     self.showImage("img", img)
        #     num = self.processImage(img, num)

        pair = []
        """ batch test """
        for (i, img) in enumerate(data):
            print("[INF0] {}/{} is selected".format(i+1, len(data)))
            # self.plotImage(1, "original image", img)
            # self.showImage("image", img)
            num = self.processImage(img, i)
            if num is not -1:
                # pair.append((i, num))
                # print("[INF0] pick ", num, "th image ")
                plt.figure(2)

                plt.subplot(2, 3, 1)
                plt.imshow(data[num], cmap='gray')

                num += 30
                if num >= 178:
                    num = num - 178

                plt.subplot(2, 3, 2)
                plt.imshow(data[num], cmap='gray')

                num += 30
                if num >= 178:
                    num = num - 178

                plt.subplot(2, 3, 3)
                plt.imshow(data[num], cmap='gray')

                num += 30
                if num >= 178:
                    num = num - 178

                plt.subplot(2, 3, 4)
                plt.imshow(data[num], cmap='gray')

                num += 30
                if num >= 178:
                    num = num - 178

                plt.subplot(2, 3, 5)
                plt.imshow(data[num], cmap='gray')

                num += 30
                if num >= 178:
                    num = num - 178

                plt.subplot(2, 3, 6)
                plt.imshow(data[num], cmap='gray')

                plt.draw()
                plt.pause(0.01)
                # self.showImage("img", data[num])
        print(pair)


if __name__ == "__main__":
    im = Image()
    im.run()
