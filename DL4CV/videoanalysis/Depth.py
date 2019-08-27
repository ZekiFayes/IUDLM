import numpy as np

# Devices.
CV_CAP_OPENNI = 900
CV_CAP_OPENNI_ASUS = 910

# Channels of an OpenNI-compatible depth generator.
# Depth map is a grayscale image in which each pixel value is the
# distance from the camera to a surface
CV_CAP_OPENNI_DEPTH_MAP = 0

# Point cloud map is a color image in which each color corresponds
# to a spatial dimension (x, y, z)
CV_CAP_OPENNI_POINT_CLOUD_MAP = 1

# disparity map is a grayscale image in which each pixel value is
# the stereo disparity of a surface
CV_CAP_OPENNI_DISPARITY_MAP = 2
CV_CAP_OPENNI_DISPARITY_MAP_32F = 3

# Valid depth mask shows whether the depth information at a given
# pixel is believed to be valid (non-zero value) or invalid (zero-value)
CV_CAP_OPENNI_VALID_DEPTH_MASK = 4

# Channels of an OpenNI-compatible RGB image generator.
CV_CAP_OPENNI_BGR_IMAGE = 5
CV_CAP_OPENNI_GRAY_IMAGE = 6


def createMedianMask(disparityMap, validDepthMask, rect=None):
    """ Return a mask selecting the median layer, plus shadows"""

    if rect is not None:
        x, y, w, h = rect
        disparityMap = disparityMap[y:y+h, x:x+w]
        validDepthMask = validDepthMask[y:y+h, x:x+w]
        median = np.median(disparityMap)

        return np.where((validDepthMask == 0) | (abs(disparityMap - median) < 12), 1.0, 0.0)
