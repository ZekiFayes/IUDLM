from imutils import paths
import pickle
import os
import cv2
import numpy as np


class DataLoader(object):

    def load(self, img_paths):
        data = []

        for (i, p) in enumerate(img_paths):
            print("[INF0] processed {}/{}".format(i + 1, len(img_paths)))
            image = cv2.imread(p, 0)
            if image is None:
                print("[INF0]", i, "th image is loaded unsuccessfully ... ")
            else:
                print("[INF0]", i, "th image is loaded successfully ... ")
                data.append(image)
        return np.array(data)

    def pkl_preprocess(self, image_paths):

        print("[INF0] loading image paths ... ")
        imagePaths = list(paths.list_images(image_paths))

        if imagePaths is None:
            print("[INFO] image paths are empty! Please check the image paths.")
        else:
            data = self.load(imagePaths)

        """ store the data in .pkl file """
        print("[INFO] storing data in {}".format("data/data.pkl"))
        with open("data/data.pkl", "wb") as fout:
            pickle.dump(data, fout)

    def storeImagePathinTxt(self, imagePaths, name):

        print("[INFO] storing image paths in {}".format(name))
        with open(name, "w") as f:
            for (j, path) in imagePaths:
                string = ''
                temp = path.split(os.path.sep)
                for (i, item) in enumerate(temp):
                    if i != len(temp) - 1:
                        string = string + temp[i] + '/'
                    else:
                        string = string + temp[i]
                if j == len(imagePaths) - 1:
                    f.write(string + '\n')
                else:
                    f.write(string)


if __name__ == "__main__":
    dl = DataLoader()
    dl.pkl_preprocess("data/img/")
    image_paths = "data/img/"
    imagePaths = list(paths.list_images(image_paths))
    dl.storeImagePathinTxt(imagePaths, "data/content.txt")
