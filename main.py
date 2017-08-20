from pipeline.pipeline import Pipeline
from pipeline.image_features import CarClassifier


def main():
    classifier = CarClassifier(car_images='data/vehicles/**/*.png',
                               non_car_images='data/non-vehicles/**/*.png')
    classifier.train()
    pipeline = Pipeline(clf=classifier)
    pipeline.load_image('test_images/test1.jpg')
    pipeline.process()


if __name__ == '__main__':
    main()
