"""
This is to demonstrate how to use ensemble learning to
improve accuracy. we use the shallownet model.
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from iudlm.preprocessor.aspectaware import AspectAwarePreprocessor
from iudlm.preprocessor.imagetoarray import ImageToArray
from iudlm.dataloader.dataloader import DataLoader
from iudlm.model.cnn import ShallowNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from imutils import paths
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
import glob


num_models = 5
print("[INF0] loading images ... ")
filename = "dataset_model_fig/dataset/train"
imagePaths = list(paths.list_images(filename))
classNames = [imagePath.split(os.path.sep)[-1].split(".")[0] for imagePath in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(32, 32)
iap = ImageToArray()
dl = DataLoader([aap, iap])
data, labels = dl.load(imagePaths, verbose=1000)

# with open("dataset/data.pkl", "rb") as fin:
#     data, labels = pickle.load(fin)
# data = data.astype("float") / 255.0
# classNames = [str(x) for x in np.unique(labels)]

trainx, testx, trainy, testy = train_test_split(data, labels,
                                                test_size=0.25,
                                                random_state=42)

lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.fit_transform(testy)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

for i in np.arange(0, num_models):

    print("[INF0] training model {}/{}".format(i+1, num_models))
    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = ShallowNet.build(32, 32, 3, 3)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    h = model.fit_generator(aug.flow(trainx, trainy, batch_size=32),
                            validation_data=(testx, testy),
                            steps_per_epoch=len(trainx) // 32,
                            epochs=5, verbose=1)

    print("[INF0] serializing model ... ")
    p = ["model/", "model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    print("[INF0] evaluating model ... ")
    p = ["model/", "model_{}.txt".format(i)]
    prediction = model.predict(testx, batch_size=32)
    report = classification_report(testy.argmax(axis=1),
                                   prediction.argmax(axis=1),
                                   target_names=classNames)
    f = open(os.path.sep.join(p), 'w')
    f.write(report)
    f.close()

    print("[INF0] plotting figure ... ")
    p = ["model/", "model_{}.png".format(i)]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 5), h.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 5), h.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 5), h.history['acc'], label="train_acc")
    plt.plot(np.arange(0, 5), h.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))

modelPaths = os.path.sep.join(["model/", "*.model"])
modelPaths = list(glob.glob(modelPaths))
print(modelPaths)
models = []

for i, modelPath in enumerate(modelPaths):
    print("[INF0] loading model {}/{}".format(i+1, len(modelPaths)))
    models.append(load_model(modelPath))

print("[INF0] evaluating ensemble .. ")
predictions = []
for m in models:
    predictions.append(m.predict(testx, batch_size=32))

predictions = np.average(predictions, axis=0)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

