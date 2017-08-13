import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def test_color_space_features():
    img = cv2.imread('test_images/test1.jpg')
    img = img[400:500, 800:950]

    image_features = ImageFeature(img)
    image_features.histogram(cv2.COLOR_RGB2HSV)
