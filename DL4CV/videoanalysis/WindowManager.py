import cv2


class WindowManager(object):

    def __init__(self, windowName, keypressCallback=None):

        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):

        return self._isWindowCreated

    def creatWindow(self):

        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):

        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):

        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvent(self):

        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:

            keycode &= 0xFF
            self.keypressCallback(keycode)
