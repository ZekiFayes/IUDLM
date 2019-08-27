from iudlm.preprocessor.simplepreprocessor import Preprocessor
from iudlm.preprocessor.imagetoarray import ImageToArray


class VideoPreprocessor(object):

    def __init__(self):

        self._p = Preprocessor(32, 32)
        self._iap = ImageToArray()

    def preprocess(self, image):

        image = image.astype("float") / 255.0
        dst = self._p.preprocess(image)
        dst = self._iap.preprocess(dst)
        dst = dst.reshape((1, 32, 32, 3))

        return dst
