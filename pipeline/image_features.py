import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

class ImageFeature(object):

    def __init__(self, img):
        self.img = img

    def color_space(self, color_space, size=(32, 32)):
        img = np.copy(self.img)
        img = cv2.cvtColor(img, color_space)
        features = cv2.resize(img, size).ravel()
        return features

    def histogram(self, color_space, size=(32, 32)):
        features = self.color_space(color_space, size)
        plt.plot(features)
        plt.show()

    def hog(self, orientations=9, ppc=(8, 8), cpb=(2, 2), viz=False,
            feature_vector=False):
        img = np.copy(self.img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return hog(img,
                   orientations=orientations,
                   pixels_per_cell=ppc,
                   cells_per_block=cpb,
                   visualise=viz,
                   feature_vector=feature_vector)

def test_color_space_features():
    img = cv2.imread('test_images/test1.jpg')
    img = img[400:500, 800:950]

    image_features = ImageFeature(img)
    features, hog = image_features.hog(viz=True, feature_vector=True)
    cv2.imwrite('output_images/hog.jpg', hog)

    # cv2.imshow('img', img)
    # cv2.imshow('hogged', hog)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # image_features.histogram(cv2.COLOR_RGB2HSV)
