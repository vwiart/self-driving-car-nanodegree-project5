import cv2
import numpy as np

class ColorSpace(object):
    """Store the params for the color space features extraction."""
    def __init__(self, color_space=None, bins=32, range=(0, 256), size=(32, 32)):
        self.bins = bins
        self.range = range
        self.color_space = color_space
        self.size = size

    def process(self, img):
        """Extract color features.
        Args:
            img: an image
            color_space: a color space from which to explore
            size: size of the image
        """
        img = np.copy(img)
        if self.color_space:
            img = cv2.cvtColor(img, self.color_space)
        spatial_features = cv2.resize(img, self.size).ravel()

        params = {
            'bins': self.bins,
            'range': self.range,
        }
        histo = lambda d: np.histogram(img[:,:,d], **params)[0]
        hist_features = np.concatenate((histo(0),
                                        histo(1),
                                        histo(2)))
        return spatial_features, hist_features
