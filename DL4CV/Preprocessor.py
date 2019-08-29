"""
This is to load and preprocess data.
We provide some options.
pickle -> .pkl
h5py   -> .hdf5
"""

from config import config
from iudlm.preprocessor.aspectaware import AspectAwarePreprocessor
from iudlm.dataloader.dataloader import DataLoader
from iudlm.preprocessor.imagetoarray import ImageToArray
from iudlm.IO.hdf5datasetwriter import HDF5DatasetWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import os
import json
import cv2
import pickle


def pkl_preprocess(filename, width, height):

    print("[INF0] loading images ... ")
    imagePaths = list(paths.list_images(filename))
    aap = AspectAwarePreprocessor(width, height)
    iap = ImageToArray()
    dl = DataLoader([aap, iap])
    data, labels = dl.load(imagePaths, verbose=1000)

    with open("dataset_model_fig/dataset/data.pkl", "wb") as fout:
        pickle.dump((data, labels), fout)


def hdf5_preprocess(filename, width, height, channel):

    print("[INF0] loading images ... ")

    imagePaths = list(paths.list_images(filename))
    classNames = [imagePath.split(os.path.sep)[-1].split(".")[0]
                  for imagePath in imagePaths]

    le = LabelEncoder()
    classNames = le.fit_transform(classNames)

    trainx, testx, trainy, testy = train_test_split(imagePaths, classNames,
                                                    test_size=config.TEST_SIZE,
                                                    stratify=classNames,
                                                    random_state=42)
    datasets = [("train", trainx, trainy, config.TRAIN_PATH),
                ("test", testx, testy, config.TEST_PATH)]

    aap = AspectAwarePreprocessor(width, height)
    (R, G, B) = ([], [], [])

    for (dType, path, label, outputPath) in datasets:
        print("[INF0] building {} ... ".format(outputPath))
        writer = HDF5DatasetWriter((len(path), width, height, channel), outputPath)

        for (p, l) in zip(path, label):
            image = cv2.imread(p)
            image = aap.preprocess(image)

            if dType == "train":
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)

            writer.add([image], [l])

        writer.close()

    print("[INF0] serializing means ... ")
    D = {
        'R': np.mean(R),
        'G': np.mean(G),
        'B': np.mean(B)
    }
    f = open(config.MEAN_PATH, "w")
    f.write(json.dumps(D))
    f.close()


if __name__ == "__main__":

    fileName = config.IMAGE_PATH
    pkl_preprocess(fileName, width=32, height=32)
    # hdf5_preprocess(fileName, width=227, height=227, channel=3)
