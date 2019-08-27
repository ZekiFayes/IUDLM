from iudlm.preprocessor.simplepreprocessor import Preprocessor
from iudlm.dataloader.dataloader import DataLoader
from iudlm.preprocessor.imagetoarray import ImageToArray
from imutils import paths
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer


def preprocess(filename):

    print("[INF0] loading image ... ")
    imagePaths = list(paths.list_images(filename))
    p = Preprocessor(32, 32)
    iap = ImageToArray()
    dl = DataLoader([p, iap])
    data, labels = dl.load(imagePaths, verbose=1000)

    with open("dataset_model_fig/dataset/data.pkl", "wb") as fout:
        pickle.dump((data, labels), fout)


if __name__ == "__main__":

    preprocess("dataset_model_fig/dataset/")
