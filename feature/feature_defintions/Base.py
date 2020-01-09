from abc import ABCMeta, abstractmethod


class FeatureBase(object, metaclass=ABCMeta):
    def __init__(self, feature_type):
        self.feature_type = feature_type

    @abstractmethod
    def get_feature(self, image):
        pass
