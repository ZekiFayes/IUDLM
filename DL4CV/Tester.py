from iudlm.preprocessor.simplepreprocessor import Preprocessor
from iudlm.preprocessor.imagetoarray import ImageToArray
from iudlm.dataloader.dataloader import DataLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2


classLabels = ["cat", "dog", "unknown"]

print("[INF0] sampling images ... ")
imagePaths = np.array(list(paths.list_images("dataset_model_fig/dataset/test1")))

idxs = np.random.randint(0, len(imagePaths), size=(1, ))
imagePaths = imagePaths[idxs]

p = Preprocessor(32, 32)
iap = ImageToArray()
dl = DataLoader(preprocessor=[p, iap])
data, labels = dl.load(imagePaths)
data = data.astype("float") / 255.0

print("[INF0] loading pre-trained model ... ")
model = load_model("dataset_model_fig/model/minivggnet.hdf5")

print("[INF0] predicting ... ")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):

    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
                (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)

