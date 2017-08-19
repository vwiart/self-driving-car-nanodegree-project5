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
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ExtractFeature(threading.Thread):
    """Extract the features from images."""

    def __init__(self, path, nbins=32, bins_range=(0, 256)):
        """Initialize the object.
        Args:
            path: directory where the images are located
            nbins: number of bins for HOG
            bins_range: range of bins for HOG
        """
        self.path = path
        self.features = []
        self.nbins=nbins
        self.bins_range=bins_range
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

        histo = lambda d: np.histogram(img[:,:,d],
                                       bins=self.nbins,
                                       range=self.bins_range)
        hist_features = np.concatenate((histo(0)[0],
                                        histo(1)[0],
                                        histo(2)[0]))
        return spatial_features, hist_features

    def _hog(self, img, orientations=9, ppc=(8, 8), cpb=(2, 2)):
        """Extract the HOG (histogram of Oriented Gradients).
        Args:
            img: an image
            orientations: number of orientations bins
            ppc: pixels per cell
            cpb: cells per block
        """
        img = np.copy(img)

        hog_features = []
        for channel in range(img.shape[2]):
            hf = hog(img[:,:, channel],
                     orientations=orientations,
                     pixels_per_cell=ppc,
                     cells_per_block=cpb, 
                     transform_sqrt=True,
                     visualise=False,
                     feature_vector=True)
            hog_features.append(hf)
        return np.ravel(hog_features)

    def _extract(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        spatial_features, hist_features = self._color_space(img)
        hog_features = self._hog(img)

        return np.concatenate((spatial_features, hist_features, hog_features))

    def run(self):
        logger.debug('[run] Extracting features for path %s' % self.path)
        images = glob(self.path)
        for filename in images:
            feat = self._extract(filename=filename)
            self.features.append(feat)


class CarClassifier(object):

    def __init__(self, test_size=0.2):
        self.classifier = LinearSVC()
        self.test_size = test_size
        self.car_features = None
        self.non_car_features = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def process_data(self, car_images, non_car_images):
        if os.path.exists(FEATURES_CHECKPOINT):
            logger.debug('[process_data] Loading features from checkpoint')
            with open(FEATURES_CHECKPOINT, mode='rb') as f:
                self.car_features, self.non_car_features = pickle.load(f)
                return 
        car_thread = ExtractFeature(car_images)
        car_thread.start()

        other_thread = ExtractFeature(non_car_images)
        other_thread.start()

        car_thread.join()
        other_thread.join()

        self.car_features = car_thread.features
        self.non_car_features = other_thread.features
        with open(FEATURES_CHECKPOINT, mode='wb') as f:
            pickle.dump((self.car_features, self.non_car_features), f)

    def train(self):
        logger.debug('[Training]')
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

        # Split dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(scaled_x, y, **options)
        self.classifier.fit(self.x_train, self.y_train)
    
    def accuracy(self):
        return self.classifier.score(self.x_test, self.y_test)
