from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class MiniVGGNet(object):

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


epoch = 100
with open("dataset/mnist.pkl", "rb") as fin:
    data, labels = pickle.load(fin)

data = data.astype("float") / 255.0
data = data.reshape((-1, 28, 28, 1))

trainx, testx, trainy, testy = \
    train_test_split(data, labels, test_size=0.25)

lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)

print("[INF0] compiling model ... ")
model = MiniVGGNet.build(28, 28, 1, 10)

print("[INF0] training network ... ")
sgd = SGD(0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
H = model.fit(trainx, trainy,
              validation_data=(testx, testy),
              epochs=epoch, batch_size=32, verbose=1)

print("[INF0] serializing network ... ")
model.save("model/minivggnet.hdf5")

print("[INF0] loading pre-training network ... ")
# model = load_model("model.hdf5")

print("[INF0] evaluating network ... ")
prediction = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            prediction.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), H.history['acc'], label="train_acc")
plt.plot(np.arange(0, epoch), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("fig/minivggnet.png")
