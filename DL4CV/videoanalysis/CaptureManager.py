import cv2
import numpy as np
import time


class CaptureManager(object):

    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):

        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFileName = None
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._frameElapsed = int(0)
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value

            # Clear the channel and frame
            self._frame = None

    # read the frame
    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve(self.channel)
            return self._frame

    @property
    def isWritingImage(self):
        return self._imageFileName is not None

    @property
    def isWritingVideo(self):
        return self._imageFileName is not None

    # enter frame
    def enterFrame(self):

        """ check that any previous frame was exited """
        assert not self._enteredFrame, 'previous enterFrame() had no matching exitFrame'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):

        if self.frame is None:
            self._enteredFrame = False

        """ update the FPS estimate and related variables """
        if self._frameElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._frameElapsed / timeElapsed

        self._frameElapsed += 1

        """ draw to the window, if any """
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        """ write to the image file, if any """
        if self.isWritingImage:
            cv2.imwrite(self._imageFileName, self._frame)
            self._imageFileName = None

        """ release the frame """
        self._frame = None
        self._enteredFrame = False

        """ write to the video file, if any """
        self._writeVideoFrame()

    def writeImage(self, filename):
        self._imageFileName = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        self._videoFileName = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                if self._frameElapsed < 20:
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFileName, self._videoEncoding,
                                                fps, size)
            self._videoWriter.write(self._frame)
