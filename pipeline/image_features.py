import logging
import os
import threading

import cv2
from glob import glob
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


FEATURES_CHECKPOINT = 'data/features.p'
MODEL_CHECKPOINT = 'data/model.p'
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ColorSpaceParams(object):
    """Store the params for the color space features extraction."""
    def __init__(self, color_space=None, bins=32, range=(0, 256)):
        self.bins = bins
        self.range = range
        self.color_space = color_space

    def get(self):
        return {
            'color_space': self.color_space,
            'bins': self.bins,
            'range': self.range,
        }


class HOGParams(object):
    """Store the params for the HOG features extraction."""
    def __init__(self, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), transform_sqrt=True):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform_sqrt = transform_sqrt
        self.visualise = False
        self.feature_vector = True

    def get(self):
        """get the HOG parameters"""
        return {
            'orientations': self.orientations,
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'transform_sqrt': self.transform_sqrt,
            'visualise': self.visualise,
            'feature_vector': self.feature_vector,
        }

class ExtractFeature(threading.Thread):
    """Extract the features from images."""

    def __init__(self, path, hog_params, color_space_params):
        """Initialize the object.
        Args:
            path: directory where the images are located
            nbins: number of bins for HOG
            bins_range: range of bins for HOG
        """
        self.path = path
        self.features = []
        self.hog_params = hog_params
        self.color_space_params = color_space_params
        super().__init__()

    def _color_space(self, img, color_space=None, size=(32, 32)):
        """Extract color features.
        Args:
            img: an image
            color_space: a color space from which to explore
            size: size of the image
        """
        img = np.copy(img)
        if color_space:
            img = cv2.cvtColor(img, color_space)

        spatial_features = cv2.resize(img, size).ravel()

        params = self.color_space_params.get()
        params.pop('color_space')
        histo = lambda d: np.histogram(img[:,:,d], **params)
        hist_features = np.concatenate((histo(0)[0],
                                        histo(1)[0],
                                        histo(2)[0]))
        return spatial_features, hist_features

    def _hog(self, img):
        """Extract the HOG (histogram of Oriented Gradients)."""
        img = np.copy(img)

        hog_features = []
        for channel in range(img.shape[2]):
            params = self.hog_params.get()
            hf = hog(img[:,:, channel], **params)
            hog_features.append(hf)
        return np.ravel(hog_features)

    def _extract(self, filename):
        """Extract the features of an image."""
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        spatial_features, hist_features = self._color_space(img)
        hog_features = self._hog(img)

        return np.concatenate((spatial_features, hist_features, hog_features))

    def run(self):
        """Loop over a set of image and extract features for each of them."""
        logger.debug('[run] Extracting features for path %s' % self.path)
        images = glob(self.path)
        for filename in images:
            feat = self._extract(filename=filename)
            self.features.append(feat)


class CarClassifier(object):

    def __init__(self, car_images, non_car_images,
                 hog_params=None, color_space_params=None, test_size=0.2):
        self.classifier = LinearSVC()
        self.car_images = car_images
        self.non_car_images = non_car_images
        self.hog_params = hog_params if hog_params else HOGParams()
        self.color_space_params = color_space_params if color_space_params else ColorSpaceParams()
        self.test_size = test_size
    
    def _process_data(self):
        """Load images and extract the features from them"""
        if os.path.exists(FEATURES_CHECKPOINT):
            logger.debug('[process_data] Loading features from checkpoint')
            with open(FEATURES_CHECKPOINT, mode='rb') as f:
                self.car_features, self.non_car_features = pickle.load(f)
                return 
        car_thread = ExtractFeature(self.car_images, self.hog_params, self.color_space_params)
        car_thread.start()

        other_thread = ExtractFeature(self.non_car_images, self.hog_params, self.color_space_params)
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
                self.classifier = pickle.load(f)
                return
        self._process_data()
        logger.debug('[train] training...')
        # Scaling data
        x = np.vstack((self.car_features, self.non_car_features)).astype(np.float64)
        x_scaler = StandardScaler().fit(x)
        scaled_x = x_scaler.transform(x)

        # Setup labels
        y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.non_car_features))))
        options = {
            'test_size': self.test_size,
            'random_state': np.random.randint(0, 100)
        }

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(scaled_x, y, **options)
        self.classifier.fit(self.x_train, self.y_train)

        accuracy = self.classifier.score(self.x_test, self.y_test)
        logger.debug('[train] accuracy = %s' % accuracy)

        with open(MODEL_CHECKPOINT, mode='wb') as f:
            logger.debug('[train] Dumping model')
            pickle.dump(self.classifier, f)
