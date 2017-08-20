import cv2
from pipeline.image_features import HOGParams
from pipeline.window import Window

class Pipeline(object):

    def __init__(self, clf, hog_params=None, img=None):
        self.classifier = clf
        self.img = img
        self.hog_params = hog_params if hog_params else HOGParams()

    def load_image(self, filename):
        self.img = cv2.imread(filename, cv2.IMREAD_COLOR)

    def process(self):
        window = Window(self.img, self.classifier)
        img = window.slide()

        cv2.imwrite('output_images/classify.png', img)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()