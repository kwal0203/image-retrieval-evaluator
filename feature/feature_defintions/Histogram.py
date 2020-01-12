from feature.feature_defintions.Base import FeatureBase
from skimage.color import rgb2gray, rgb2hsv
from sklearn.preprocessing import normalize

import numpy as np


class Histogram(FeatureBase):
    def __init__(self, bins, config):
        super().__init__(config=config)
        self.bins = bins

    # 1 channel grayscale histogram feature
    def get_l_feature(self, image):
        # Convert image to 8-bit gray-scale
        image = np.uint8(rgb2gray(image) * 255)

        # Create "self.bin" sized normalized histogram of image
        # are we using 64-bit floats here?
        # Note: np.histogram() returns a tuple and the first element is the
        # histogram
        gray_histogram = np.histogram(image, self.bins, (0, self.bins))
        gray_histogram = normalize(gray_histogram[0].reshape(1, -1)).flatten()
        assert(gray_histogram.size == self.bins)
        return gray_histogram

    # 2 channel Hue-Value histogram feature
    def get_hv_feature(self, image):
        # Convert image to 8-bit Hue-Saturation-Value format
        image = np.uint8(rgb2hsv(image) * 255)

        # Create "self.bin" sized normalized histograms for Hue and Value
        # channels
        # Note: np.histogram() returns a tuple and the first element is the
        # histogram
        hue_histogram = np.histogram(image[:, :, 0], self.bins, (0, self.bins))
        val_histogram = np.histogram(image[:, :, 2], self.bins, (0, self.bins))

        # Concatenate Hue and Value histograms to get the HV histogram feature
        # vector and then normalize
        hv_histogram = np.concatenate((hue_histogram[0], val_histogram[0]))
        hv_histogram = normalize(hv_histogram.reshape(1, -1)).flatten()
        assert(hv_histogram.size == self.bins * 2)
        return hv_histogram

    # 3 channel Red-Green-Blue histogram feature
    def get_rgb_feature(self, image):
        # Create "self.bin" sized normalized histograms for Red, Green and Blue
        # channels
        # Note: np.histogram() returns a tuple and the first element is the
        # histogram
        red_histogram = np.histogram(image[:, :, 0], self.bins, (0, self.bins))
        grn_histogram = np.histogram(image[:, :, 1], self.bins, (0, self.bins))
        blu_histogram = np.histogram(image[:, :, 2], self.bins, (0, self.bins))

        # Concatenate Red, Green and Blue histograms to get the RGB histogram
        # feature vector
        rgb_histogram = np.concatenate((red_histogram[0], grn_histogram[0]))
        rgb_histogram = np.concatenate((rgb_histogram, blu_histogram[0]))
        rgb_histogram = normalize(rgb_histogram.reshape(1, -1)).flatten()
        assert(rgb_histogram.size == self.bins * 3)
        return rgb_histogram

    # Determine which function to call depending on feature user has requested
    def get_feature(self, image):
        if "gray" in self.feature_name:
            feature = self.get_l_feature(image=image)
        elif "hv" in self.feature_name:
            feature = self.get_hv_feature(image=image)
        elif "rgb" in self.feature_name:
            feature = self.get_rgb_feature(image=image)
        else:
            feature = None
            print("Feature '{}' not valid".format(self.feature_name))

        return feature
