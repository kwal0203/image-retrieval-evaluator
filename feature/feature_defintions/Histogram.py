from feature.feature_defintions.Base import FeatureBase


class Histogram(FeatureBase):
    def __init__(self, bins, histogram_type):
        self.bins = bins
        self.histogram_type = histogram_type

    # 1 channel grayscale histogram feature
    def get_l_feature(self):
        print("{} feature not implemented ".format(self.histogram_type))

    # 2 channel Hue-Value histogram feature
    def get_hv_feature(self):
        print("{} feature not implemented ".format(self.histogram_type))

    # 3 channel Red-Green-Blue histogram feature
    def get_rgb_feature(self):
        print("{} feature not implemented ".format(self.histogram_type))

    def get_feature(self, image):
        # Determine type of histogram
        # Create histogram for each channel of size "bins"
        # Concatenate channel histograms if necessary
        # Normalize concatenated feature
        # Return feature
        return [1, 2, 3, 4, 5]
