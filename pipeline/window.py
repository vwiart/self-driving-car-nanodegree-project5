import numpy as np

import cv2
from pipeline.extract_features import ExtractFeatures
from pipeline.hog import HOG
from pipeline.color_space import ColorSpace

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

    def __init__(self, img, clf,
                 width=16, height=16,
                 hog=None, color_space=None):
        self.img = img
        self.classifier = clf
        self.hog = hog if hog else HOG()
        self.color_space = color_space if color_space else ColorSpace()
        self.width = width
        self.height = height

    def classify(self, top_left, bottom_right):
        box = Box(top_left, bottom_right, border=2)
        crop = box.crop(self.img)

        color_space = ColorSpace()
        feat_extractor = ExtractFeatures(self.hog, color_space)
        # features = feat_extractor.extract(img=crop, box=box)
        features = extract_features(crop, self.color_space, self.hog)

        is_car = self.classifier.predict(features)
        if is_car:
            self.img = box.draw(self.img, color=(0, 255, 0))

    def slide(self):
        h, w, _ = self.img.shape
        box = Box(Point(0, h//2), Point(w, h))
        self.img = box.crop(self.img)
        max_height, max_width, _ = self.img.shape

        width = max_width//self.width
        height = max_height// self.height

        self.hog.process(self.img)
        for x in range(width):
            for y in range(height):
                top_left = Point(x * self.width, y * self.height)
                bottom_right = Point(x * self.width + 64, y * self.height + 64)
                if bottom_right.x > max_width or bottom_right.y > max_height:
                    continue
                self.classify(top_left, bottom_right)

        return self.img
