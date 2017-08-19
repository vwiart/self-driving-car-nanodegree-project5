from pipeline.image_features import CarClassifier


def train_classifier():
    classifier = CarClassifier()
    classifier.process_data(car_images='data/vehicles/**/*.png',
                            non_car_images='data/non-vehicles/**/*.png')
    classifier.train()
    print(classifier.accuracy())
