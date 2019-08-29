"""
This is to demonstrate how to fine-tune a pre-trained model.
Here, we take VGG16 as an example.
"""


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from iudlm.preprocessor.aspectaware import AspectAwarePreprocessor
from iudlm.preprocessor.imagetoarray import ImageToArray
from iudlm.dataloader.dataloader import DataLoader
from iudlm.model.fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import os


print("[INF0] loading images ... ")
filename = "dataset_model_fig/dataset/train"
imagePaths = list(paths.list_images(filename))
classNames = [imagePath.split(os.path.sep)[-1].split(".")[0] for imagePath in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(32, 32)
iap = ImageToArray()
dl = DataLoader([aap, iap])
data, labels = dl.load(imagePaths, verbose=1000)
data = data.astype("float") / 255.0

trainx, testx, trainy, testy = train_test_split(data, labels,
                                                test_size=0.25,
                                                random_state=42)

lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.fit_transform(testy)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True,
                         fill_mode="nearest")

baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(32, 32, 3)))

headModel = FCHeadNet.build(baseModel, len(classNames), 256)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INF0] compiling model ... ")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INF0] training head ... ")
model.fit_generator(aug.flow(trainx, trainy, batch_size=32),
                    validation_data=(testx, testy), epochs=5,
                    steps_per_epoch=len(trainx) // 32, verbose=1)

print("[INF0] evaluating after initialization ... ")
predictions = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

for layer in baseModel.layers[15:]:
    layer.trainable = True

print("[INF0] re-compiling model ... ")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INF0] training head ... ")
model.fit_generator(aug.flow(trainx, trainy, batch_size=32),
                    validation_data=(testx, testy), epochs=5,
                    steps_per_epoch=len(trainx) // 32, verbose=1)

print("[INF0] evaluating after initialization ... ")
predictions = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

print("[INF0] serializing model ... ")
model.save("model/m_vgg.hdf5")
