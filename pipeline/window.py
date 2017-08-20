import cv2
import numpy as np

from pipeline.image_features import ColorSpaceParams, ExtractFeatures, HOGParams

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def as_tuple(self):
        return (self.x, self.y)


class Box(object):

    def __init__(self, top_left, bottom_right, border=6):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.border = border

    def draw(self, img, color=(0, 0, 255)):
        img = np.copy(img)
        cv2.rectangle(img,
                      self.top_left.as_tuple(),
                      self.bottom_right.as_tuple(),
                      color,
                      self.border)
        return img

    def crop(self, img):
        img = np.copy(img)
        x1, y1 = self.top_left.x, self.top_left.y
        x2, y2 = self.bottom_right.x, self.bottom_right.y
        return img[y1:y2, x1:x2]


class Window(object):

    def __init__(self, img, clf):
        self.img = img
        self.classifier = clf

    def classify(self, top_left, bottom_right):
        box = Box(top_left, bottom_right, border=2)
        crop = box.crop(self.img)

        hog_params = HOGParams()
        color_space_params = ColorSpaceParams()
        feat_extractor = ExtractFeatures(hog_params, color_space_params)
        features = feat_extractor.extract(crop)
        is_car = self.classifier.predict(features)
        color = (0, 255, 0) if is_car else (0, 0, 255)
        self.img = box.draw(self.img, color)

    def slide(self):
        print(self.img.shape)
        self.classify(Point(0, 0), Point(64, 64))
        self.classify(Point(832, 384), Point(896, 448))
        
        return self.img