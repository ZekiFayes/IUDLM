from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):

    def __init__(self, figPath, jsonPath=None, startAt=0):

        super(TrainingMonitor, self).__init__()
        self._figPath = figPath
        self._jsonPath = jsonPath
        self._startAt = startAt

    def on_train_begin(self, logs={}):

        self._H = {}

        if self._jsonPath is not None:
            if os.path.exists(self._jsonPath):
                self._H = json.loads(open(self._jsonPath).read())

                if self._startAt > 0:
                    for k in self._H.keys():
                        self._H[k] = self._H[k][:self._startAt]

    def on_epoch_end(self, epoch, logs={}):

        for (k, v) in logs.items():
            l = self._H.get(k, [])
            l.append(v)
            self._H[k] = l

        if self._jsonPath is not None:
            f = open(self._jsonPath, "w")
            f.write(json.dumps(self._H))
            f.close()

        if len(self._H["loss"]) > 1:

            N = np.arange(0, len(self._H["loss"]))
            plt.style.use("ggplot")
            plt.figure()

            plt.plot(N, self._H["loss"], label="train_loss")
            plt.plot(N, self._H["val_loss"], label="val_loss")
            plt.plot(N, self._H['acc'], label="train_acc")
            plt.plot(N, self._H["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(self._figPath)
            plt.close()