import cv2
from videoanalysis.CaptureManager import CaptureManager
from videoanalysis.WindowManager import WindowManager
from keras.models import load_model
from videoanalysis.VideoPreprocessor import VideoPreprocessor


class Cameo(object):

    def __init__(self):

        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, False)
        self._model = load_model("dataset_model_fig/model/minivggnet.hdf5")
        self._preprocessor = VideoPreprocessor()
        self._classLabels = ["cat", "dog", "unknown"]

    def run(self):

        self._windowManager.creatWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            image = frame.copy()
            image = self._preprocessor.preprocess(image)

            pred = int(self._model.predict(image, batch_size=32).argmax(axis=1))
            cv2.putText(frame, "Label: {}".format(self._classLabels[pred]),
                        (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self._windowManager.show(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvent()

    def onKeypress(self, keycode):

        """
        space  -> take a screen shot
        tab    -> start/stop recording a screencast
        escape -> quit
        """

        # space
        if keycode == 32:
            self._captureManager.writeImage('screenshot.png')

        # tab
        elif keycode == 9:
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        # escape
        elif keycode == 27:
            self._windowManager.destroyWindow()
