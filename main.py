import cv2
from glob import glob
# from pipeline.pipeline import Pipeline, process
from pipeline.pipeline import process

from pipeline.classifier import CarClassifier
from pipeline.color_space import ColorSpace
from pipeline.hog import HOG
from moviepy.editor import VideoFileClip


def main():
    images = ['test_images/test1.jpg']
    # images = ['data/vehicles/GTI_Far/image0001.png',
    #           'data/non-vehicles/Extras/extra1.png',
    #           'data/non-vehicles/Extras/extra2.png',
    #           'data/non-vehicles/Extras/extra3.png']
    hog = HOG()
    # cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2LUV, cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2YUV
    # cv2.COLOR_RGB2YCrCb
    color_space = ColorSpace(color_space=cv2.COLOR_RGB2HSV, size=(64, 64))
    # color_space = ColorSpace(color_space=cv2.COLOR_RGB2YCrCb)
    classifier = CarClassifier(car_images='data/vehicles/**/*.png',
                               non_car_images='data/non-vehicles/**/*.png',
                               hog=hog,
                               color_space=color_space)
    classifier.train()
    for filename in images:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = process(img, classifier, color_space, hog)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
