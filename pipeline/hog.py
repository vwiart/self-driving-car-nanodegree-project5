import numpy as np
from skimage.feature import hog

class HOG(object):
    """Store the params for the HOG features extraction."""
    def __init__(self, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), transform_sqrt=True):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform_sqrt = transform_sqrt
        self.visualise = False
        self.feature_vector = True
        self.features = None

    def process(self, img):
        """Extract the HOG (histogram of Oriented Gradients)."""
        if self.features is not None:
            return self.features

        img = np.copy(img)

        self.features = []
        for channel in range(img.shape[2]):
            params = {
                'orientations': self.orientations,
                'pixels_per_cell': self.pixels_per_cell,
                'cells_per_block': self.cells_per_block,
                'transform_sqrt': self.transform_sqrt,
                'visualise': self.visualise,
                'feature_vector': self.feature_vector,
            }
            hf = hog(img[:,:, channel], **params)
            self.features.append(hf)
        return self.features
