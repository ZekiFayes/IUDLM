import numpy as np
import cv2
import os


class DataLoader(object):

    def __init__(self, preprocessor):

        self.preprocessor = preprocessor

        if self.preprocessor is None:
            self.preprocessor = []

    def load(self, imagePaths, verbose=-1):

        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):

            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-1].split(".")[0]

            if self.preprocessor is not None:
                for p in self.preprocessor:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INF0] processed {}/{}".format(i + 1, len(imagePaths)))

        return np.array(data), np.array(labels)
