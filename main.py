from pipeline import pipeline
from pipeline.image_features import CarClassifier


def main():
    classifier = CarClassifier(car_images='data/vehicles/**/*.png',
                               non_car_images='data/non-vehicles/**/*.png')
    classifier.train()

    pipeline.process_image()


if __name__ == '__main__':
    main()
