"""
This is to demonstrate how to use hdf5 to store large dataset.
"""


import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from iudlm.IO.hdf5datasetwriter import HDF5DatasetWriter
import json
from iudlm.preprocessor.aspectaware import AspectAwarePreprocessor
from imutils import paths
import os
from sklearn.preprocessing import LabelEncoder


print("[INF0] loading images ... ")
filename = "dataset_model_fig/dataset/train"
imagePaths = list(paths.list_images(filename))
classNames = [imagePath.split(os.path.sep)[-1].split(".")[0] for imagePath in imagePaths]

le = LabelEncoder()
classNames = le.fit_transform(classNames)

trainx, testx, trainy, testy = train_test_split(imagePaths, classNames,
                                                test_size=6000, stratify=classNames,
                                                random_state=42)
datasets = [("train", trainx, trainy, "dataset_model_fig/dataset/train.hdf5"),
            ("test", testx, testy, "dataset_model_fig/dataset/test.hdf5")]

aap = AspectAwarePreprocessor(32, 32)
(R, G, B) = ([], [], [])

for (dType, path, label, outputPath) in datasets:
    print("[INF0] building {} ... ".format(outputPath))
    writer = HDF5DatasetWriter((len(path), 32, 32, 3), outputPath)

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
f = open("dataset/mean.json", "w")
f.write(json.dumps(D))
f.close()
