import cv2
from pipeline.classifier import CarClassifier
from pipeline.color_space import ColorSpace
from pipeline.hog import HOG
from pipeline.window import Window, Box, Point

import numpy as np

class Pipeline(object):

    def __init__(self, clf, hog=None, color_space=None, img=None):
        self.classifier = clf
        self.img = img
        self.hog = hog if hog else HOG()
        self.color_space = color_space if color_space else ColorSpace()

    def load_image(self, filename):
        self.img = cv2.imread(filename, cv2.IMREAD_COLOR)

    def process(self, img=None):
        img = self.img if img is None else img

        window = Window(img,
                        self.classifier,
                        hog=self.hog,
                        color_space=self.color_space)

        img = window.slide()
        return img



def extract_features(img, color_space, hog):
    hog_features = np.ravel(hog.process(img))
    spatial_features, hist_features = color_space.process(img)

    return np.concatenate((spatial_features, hist_features, hog_features))


def region_of_interest(img, width, height):
    return img[height//2:height, 0:width]


def test_process(img, classifier, color_space, hog):
    img_height, img_width, _ = img.shape

    features = extract_features(img, color_space, hog)
    is_car = classifier.predict(features)
    if is_car:
        print('is a car')


def process(img, classifier, color_space, hog):
    img_height, img_width, _ = img.shape
    roi = region_of_interest(img, img_width, img_height)
    roi_height, roi_width, _ = roi.shape
    for x in range(roi_width//32):
        for y in range(roi_height//32):
            x1, y1 = x * 32, y * 32
            x2, y2 = x * 32 + 64, y * 32 + 64
            if x2 > img_width or y2 > img_height//2:
                continue
            crop = roi[y1:y2, x1:x2]

            features = extract_features(crop, color_space, hog)
            is_car = classifier.predict(features)
            if is_car:
                cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img