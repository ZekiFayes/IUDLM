import cv2
import numpy as np
from videoanalysis import Utils


class VFuncFilter(object):
    def __init__(self, vFunc=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        self._vLookupArray = Utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        srcFlatView = Utils.createFlatView(src)
        dstFlatView = Utils.createFlatView(dst)
        Utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)


class VCurveFilter(VFuncFilter):
    def __init__(self, vPoints, dtype=np.uint8):
        VFuncFilter.__init__(self, Utils.createCurveFunc(vPoints), dtype)


class BGRFuncFilter(object):
    """A filter that applies different functions to each of BGR."""

    def __init__(self, vFunc = None, bFunc = None, gFunc = None, rFunc = None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        self._bLookupArray = Utils.createLookupArray(Utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = Utils.createLookupArray(Utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = Utils.createLookupArray(Utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src, dst):
        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        Utils.applyLookupArray(self._bLookupArray, b, b)
        Utils.applyLookupArray(self._gLookupArray, g, g)
        Utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curves to each of BGR."""
    def __init__(self, vPoints = None, bPoints = None, gPoints = None, rPoints = None, dtype=np.uint8):
        BGRFuncFilter.__init__(self, Utils.createCurveFunc(vPoints), Utils.createCurveFunc(bPoints),
                               Utils.createCurveFunc(gPoints), Utils.createCurveFunc(rPoints), dtype)


# Kodak
class BGRPortraCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""
    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(self,
                                vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
                                bPoints=[(0, 0), (41, 46), (231, 228), (255, 255)],
                                gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
                                rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)],
                                dtype=dtype)


# Fuji Provia
class BGRProviaCurveFilter(BGRCurveFilter):
    """A filter that applies Provia-like curves to BGR."""
    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(self,
                                bPoints=[(0, 0), (35, 25), (205, 227), (255, 255)],
                                gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
                                rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
                                dtype=dtype)


# Fuji Velvia
class BGRVelviaCurveFilter(BGRCurveFilter):
    """A filter that applies Velvia-like curves to BGR."""
    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(self,
                                vPoints=[(0, 0), (128, 118), (221, 215), (255, 255)],
                                bPoints=[(0, 0), (25, 21), (122, 153), (165, 206), (255, 255)],
                                gPoints=[(0, 0), (25, 21), (95, 102), (181, 208), (255, 255)],
                                rPoints=[(0, 0), (41, 28), (183, 209), (255, 255)],
                                dtype=dtype)


# cross processing
class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """A filter that applies cross-process-like curves to BGR."""
    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(self,
                                bPoints=[(0, 20), (255, 235)],
                                gPoints=[(0, 0), (56, 39), (208, 226), (255, 255)],
                                rPoints=[(0, 0), (56, 22), (211, 255), (255, 255)],
                                dtype=dtype)


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius."""
    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)
