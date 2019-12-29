from dv_interface import RunTime
import cv2
import numpy as np


if __name__ == "__main__":

    # image = cv2.imread("data/EndPlane/img1.png", 0)
    # RunTime().run(image, "end_plane")

    # image = cv2.imread("data/sealingRing/img1.png", 0)
    # RunTime().run(image, "sealing_ring")

    # image = cv2.imread("data/fillet/img1.bmp", 0)
    # RunTime().run(image, "fillet")

    # image = cv2.imread("data/innerplane/img4.png", 0)
    # RunTime().run(image, "inner_plane")

    # image = cv2.imread("data/outerplane/img2.png", 0)
    # RunTime().run(image, "outer_plane")

    image = cv2.imread("data/concave_vex/img4.png", 0)
    RunTime().run(image, "concave_vex")



