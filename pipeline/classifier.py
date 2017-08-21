import logging
import os
import threading
import time

import cv2
from glob import glob
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pipeline.extract_features import ExtractFeatures
from pipeline.hog import HOG
from pipeline.color_space import ColorSpace
from pipeline import pipeline

FEATURES_CHECKPOINT = 'data/features.p'
MODEL_CHECKPOINT = 'data/model.p'
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CarClassifier(object):

    def __init__(self, car_images, non_car_images, hog, color_space,
                 test_size=0.2):
        self.car_images = car_images
        self.non_car_images = non_car_images
        self.hog = hog
        self.color_space = color_space
        self.test_size = test_size
    
    def _process_data(self):
        """Load images and extract the features from them"""
        if os.path.exists(FEATURES_CHECKPOINT):
            logger.debug('[process_data] Loading features from checkpoint')
            with open(FEATURES_CHECKPOINT, mode='rb') as f:
                self.car_features, self.non_car_features = pickle.load(f)
                return 
        car_thread = ExtractFeatures(dataset=self.car_images,
                                     hog=self.hog,
                                     color_space=self.color_space)
        car_thread.start()

        other_thread = ExtractFeatures(dataset=self.non_car_images,
                                       hog=self.hog,
                                       color_space=self.color_space)
        other_thread.start()

        car_thread.join()
        other_thread.join()

        self.car_features = car_thread.features
        self.non_car_features = other_thread.features
        with open(FEATURES_CHECKPOINT, mode='wb') as f:
            pickle.dump((self.car_features, self.non_car_features), f)

    def train(self):
        """Train the model with a set of cars and non cars."""
        if os.path.exists(MODEL_CHECKPOINT):
            logger.debug('[train] Loading model from checkpoint')
            with open(MODEL_CHECKPOINT, mode='rb') as f:
                data = pickle.load(f)
                self.classifier = data['classifier']
                self.x_scaler = data['scaler']
                return
        self._process_data()
        logger.debug('[train] training...')
        # Scaling data
        x = np.vstack((self.car_features, self.non_car_features)).astype(np.float64)
        self.x_scaler = StandardScaler().fit(x)
        scaled_x = self.x_scaler.transform(x)

        # Setup labels
        y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.non_car_features))))
        options = {
            'test_size': self.test_size,
            'random_state': np.random.randint(0, 100),
        }

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(scaled_x, y, **options)
        self.classifier = LinearSVC()
        self.classifier.fit(self.x_train, self.y_train)

        accuracy = self.classifier.score(self.x_test, self.y_test)
        logger.debug('[train] accuracy = %s' % accuracy)

        with open(MODEL_CHECKPOINT, mode='wb') as f:
            logger.debug('[train] Dumping model')
            pickle.dump({
                'classifier': self.classifier,
                'scaler': self.x_scaler,
            }, f)

    def predict(self, features):
        reshaped = np.array(features).reshape(1, -1)
        scaled_features = self.x_scaler.transform(reshaped)
        return self.classifier.predict(scaled_features)