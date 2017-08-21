import logging
import threading

import cv2
from glob import glob

from pipeline import pipeline

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ExtractFeatures(threading.Thread):
    """Extract the features from images."""

    def __init__(self, hog, color_space, dataset=None):
        """Initialize the object.
        Args:
            dataset: directory where the images are located
            nbins: number of bins for HOG
            bins_range: range of bins for HOG
        """
        self.dataset = dataset
        self.features = []
        self.hog = hog
        self.color_space = color_space
        super().__init__()

    def run(self):
        """Bulk load a set of image and extract features for each of them."""
        logger.debug('[run] Extracting features for path %s' % self.dataset)
        images = glob(self.dataset)
        for filename in images:
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            features = pipeline.extract_features(img, self.color_space, self.hog)
            self.features.append(features)
