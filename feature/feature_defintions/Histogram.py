import numpy as np

from feature.feature_defintions.Base import FeatureBase
from skimage.color import rgb2gray, rgb2hsv


class Histogram(FeatureBase):
    def __init__(self, bins, histogram_type):
        super().__init__(histogram_type)
        self.bins = bins

    # 1 channel grayscale histogram feature
    def get_l_feature(self, image):
        # Convert image to 8-bit gray-scale
        image = rgb2gray(image)
        image = np.uint8(image * 255.0)

        # Create normalized histogram of size "bins"
        # are we using 64-bit floats here?
        gray_histogram = np.histogram(a=image, bins=self.bins, density=True)[0]
        assert(gray_histogram.size == 256)

        return gray_histogram

    # 2 channel Hue-Value histogram feature
    def get_hv_feature(self, image):
        # Convert image to 8-bit Hue-Saturation-Value format
        image = rgb2hsv(image)
        image = np.uint8(image * 255)

        # Create normalized histograms for Hue and Value channels of size "bins"
        hue_histogram = np.histogram(a=image[0], bins=self.bins, density=True)[0]
        value_histogram = np.histogram(a=image[2], bins=self.bins, density=True)[0]

        # Concatenate Hue and Value histograms to get the HV histogram feature
        # vector
        hv_histogram = np.concatenate((hue_histogram, value_histogram), axis=None)
        assert(hv_histogram.size == 512), "Size is {}".format(hv_histogram.size)

        return hv_histogram

    # 3 channel Red-Green-Blue histogram feature
    def get_rgb_feature(self, image):
        # Create normalized histograms for Red, Green and Blue channels of size
        # "bins"
        red_histogram = np.histogram(a=image[0], bins=self.bins, density=True)[0]
        grn_histogram = np.histogram(a=image[1], bins=self.bins, density=True)[0]
        blue_histogram = np.histogram(a=image[2], bins=self.bins, density=True)[0]

        # Concatenate Red, Green and Blue histograms to get the RGB histogram
        # feature vector
        rgb_histogram = np.concatenate((red_histogram, grn_histogram), axis=None)
        rgb_feature = np.concatenate((rgb_histogram, blue_histogram), axis=None)
        assert(rgb_feature.size == 768)

        return rgb_feature

    # Determine which function to call depending on feature user has requested
    def get_feature(self, image):
        if "gray" in self.feature_type:
            print("Gray histogram selected with {} bins".format(self.bins))
            feature = self.get_l_feature(image=image)
        elif "hv" in self.feature_type:
            print("HV histogram selected with {} bins".format(self.bins))
            feature = self.get_hv_feature(image=image)
        elif "rgb" in self.feature_type:
            print("RGB histogram selected with {} bins".format(self.bins))
            feature = self.get_rgb_feature(image=image)
        else:
            feature = None
            print("Feature '{}' not valid".format(self.feature_type))

        return feature
