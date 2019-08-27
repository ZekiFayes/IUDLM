from iudlm.model.cnn import ShallowNet
from iudlm.model.lenet import LeNet
from iudlm.model.minivggnet import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt


class IUDLM(object):

    def __init__(self, model, name):

        self._model = model
        self._name = name
        self._lb = LabelBinarizer()
        self._trainx = None
        self._testx = None
        self._trainy = None
        self._testy = None
        self._h = None
        self._epoch = 5

    def loaddata(self):

        with open("dataset_model_fig/dataset/data.pkl", "rb") as fin:
            data, labels = pickle.load(fin)

        """ normalization is important. In practice, it can improve the accuracy """
        data = data.astype("float") / 255.0
        return data, labels

    def splitdata(self, data, labels):

        labels = self._lb.fit_transform(labels)
        self._trainx, self._testx, self._trainy, self._testy = train_test_split(
            data, labels, test_size=0.25, random_state=42)

    def train(self):

        print("[INF0] training model ... ")
        opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
        self._model.compile(loss="categorical_crossentropy",
                            optimizer=opt, metrics=["accuracy"])
        self._h = self._model.fit(self._trainx, self._trainy,
                                  validation_data=(self._testx, self._testy),
                                  epochs=self._epoch, batch_size=128, verbose=1)

        print("[INF0] serializing model ... ")
        self._model.save("dataset_model_fig/model/" + self._name + ".hdf5")

    def evaluate(self):

        print("[INF0] evaluating model ... ")
        prediction = self._model.predict(self._testx, batch_size=32)
        print(classification_report(self._testy.argmax(axis=1),
                                    prediction.argmax(axis=1),
                                    target_names=[str(x) for x in self._lb.classes_]))

    def plot(self):

        print("[INF0] plotting figure ... ")
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self._epoch), self._h.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self._epoch), self._h.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self._epoch), self._h.history['acc'], label="train_acc")
        plt.plot(np.arange(0, self._epoch), self._h.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig("dataset_model_fig/fig/" + self._name + ".png")

    def run(self):

        data, labels = self.loaddata()
        self.splitdata(data, labels)
        self.train()
        self.plot()
        self.evaluate()


if __name__ == "__main__":

    m1 = ShallowNet.build(32, 32, 3, 3)
    m2 = LeNet.build(32, 32, 3, 3)
    m3 = MiniVGGNet.build(32, 32, 3, 3)
    M = [m1, m2, m3]
    names = ["cnn", "lenet", "minivggnet"]

    for m, n in zip(M, names):
        iudlm = IUDLM(m, n)
        iudlm.run()
