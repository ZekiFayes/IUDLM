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

        return cv2.imread(name)

    def binarizeImage(self, image):

        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    def getImageShape(self, src):

        return src.shape[:2]

    def getOuterEdegPoints(self, src):

        pts = []
        h, w = self.getImageShape(src)
        for i in range(w):
            for j in range(h):
                if src[j, i] > 200:
                    pts.append([i, j])
                    break
        return pts

    def getInnerEdgePoints(self, src):

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

        pts_sorted = sorted(pts)
        pt1 = pts_sorted[0]
        pt2 = pts_sorted[-1]
        rightAngle = math.atan2((pt1[1] - center[1]), (pt1[0] - center[0]))
        leftAngle = math.atan2((pt2[1] - center[1]), (pt2[0] - center[0]))

        return -leftAngle, -rightAngle

    def cart2polar(self, src, center, radius):

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

        x = center[0] + point[1] * np.cos(point[0] / radius)
        y = center[1] + point[1] * np.sin(point[0] / radius)

        return [y, x]

    def shiftImage(self, src, theta_offset):

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

        resized = cv2.resize(src, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(name, resized)
        cv2.waitKey(0)

    def findContours(self, src):

        contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def findBoundingBoxes(self, contours):

        if contours is None:
            return []

        rect_boxes = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            # if 20 < w < 80 and 30 < h < 70:
            if 10 < w < 1000:
                rect_boxes.append((x, y, w, h))

        return rect_boxes

    def plotString(self, src, boxes):

        if boxes is None:
            print("there is no box!")
            return

        i = 0
        for (x, y, w, h) in boxes:
            roi = src[y:y+h, x:x+w]
            cv2.imshow(str(i), roi)
            cv2.waitKey(0)
            i += 1

    def detectString(self, boxes):

        if boxes is None:
            print("there is no box!")
            return 0, [], []

        box1 = []
        box2 = []

        if not len(boxes):
            print("there is no box!")
            return 0, [], []
        elif len(boxes) == 1:
            return 1, boxes, []
        else:
            boxes = np.array(boxes)
            mean = np.mean(boxes, 0)
            print("mean = ", mean)
            std = np.std(boxes, 0, ddof=1)
            print("std = ", std)

            if std[0] > 500:
                for (x, y, w, h) in boxes:
                    if x < mean[0]:
                        box1.append((x, y, w, h))
                    else:
                        box2.append((x, y, w, h))

                print("2 classes")
                return 2, box1, box2
            else:
                print("1 class")
                return 1, boxes, []

    def processImage(self, src, ith_image):

        img = src[20:1060, 20:1900]

        print("[INF0] binarizing image ... ")
        binarized_images = self.binarizeImage(img)

        print("[INF0] fitting circle ... ")
        pts = self.getOuterEdegPoints(binarized_images)
        center, rMax = self.getCenterandRadius(pts)

        print("[INF0] calculating angle range ... ")
        theta_min, theta_max = self.getAngleRange(pts, center, rMax)
        print("theta_min = ", theta_min)
        print("theta_max = ", theta_max)

        pts = self.getInnerEdgePoints(binarized_images)
        center1, rMin = self.getCenterandRadius(pts)

        print("[INF0] transforming image ... ")
        cart_image = self.cart2polar(img, center, rMin, rMax, theta_min, theta_max)
        # self.showImage("cart_image", cart_image)

        mean = np.mean(cart_image)

        print("[INF0] thresholding image ... ")
        _, thresh = cv2.threshold(cart_image, int(mean), 255, cv2.THRESH_BINARY_INV)
        roi = thresh[0:140]
        self.showImage("roi", roi)

        print("[INF0] finding contours ... ")
        contours = self.findContours(roi)

        if contours is not None:
            print("[INF0] finding boxes ... ")
            boxes = self.findBoundingBoxes(contours)

            print("[INF0] classifying boxes ... ")
            num_classes, box1, box2 = self.detectString(boxes)

            if num_classes == 1:
                self.plotString(roi, box1)
                xywh = np.array(box1)
                mean = np.mean(xywh, 0)
                num = int((85 + (mean[0] + mean[2] / 2) / ((theta_max - theta_min) * rMax) * 180 / np.pi) / 2)
                print(num)

                if ith_image + num > 178:
                    num = ith_image + num - 178
                else:
                    num = num + ith_image

                return num

            elif num_classes == 2:
                self.plotString(roi, box1)
                self.plotString(roi, box2)

                return -1
            else:
                return -1

    def run(self):

        with open("data/data.pkl", "rb") as fin:
            data = pickle.load(fin)

        ith_image = np.random.randint(0, 177)
        print("ith_image =", ith_image)
        img = data[ith_image]
        self.showImage("image", img)
        num = self.processImage(img, ith_image)

        if num is not -1:
            print("pick ", num, "th image ")
            img = data[num]
            self.showImage("img", img)


if __name__ == "__main__":
    im = Image()
    im.run()
    # im = cv2.imread("data/img/img_1.jpg", 0)

    # with open("data/data.pkl", "rb") as fin:
    #     data = pickle.load(fin)
    #
    # find_starting_point = False
    # ith_image = 0
    # while True:
    #     # read image
    #     if find_starting_point is False:
    #         img = data[ith_image]
    #
    #         thresh = im.binarizeImage(img)
    #         pts = im.getInnerEdgePoints(thresh)
    #         center, radius = im.getCenterandRadius(pts)
    #         cart_image = im.cart2polar(img, center, radius / 45 * 80)
    #         thresh = im.binarizeImage(cart_image)
    #         thresh = thresh[int(radius):, 1375:3550]
    #
    #         im.showImage("image", thresh)
    #         break

    #
    # for (i, item) in enumerate(data):
    #     temp = item[850: 1050, 1200:3800]
    #     kernel = np.ones((3, 3), np.uint8)
    #     temp = cv2.flip(temp, 0)
    #     temp = cv2.bitwise_not(temp)
    #     temp = cv2.dilate(temp, kernel, iterations=3)
    #     temp = cv2.erode(temp, kernel, iterations=3)
    #
    #     cnts, heirarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     boxes = []
    #     for (j, c) in enumerate(cnts):
    #         (x, y, w, h) = cv2.boundingRect(c)
    #         boxes.append((x, y, w, h))
    #
    #     if boxes is not None:
    #         for (j, (x, y, w, h)) in enumerate(boxes):
    #             if 50 < h < 70 and 20 < w < 90:
    #                 print((x, y, w, h))
    #                 roi = temp[y - 5: y + h + 5, x - 5: x + w + 5]
    #                 plt.subplot(3, 3, j+1)
    #                 plt.title(str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h))
    #                 plt.imshow(roi, cmap='gray')
    #         plt.show()
