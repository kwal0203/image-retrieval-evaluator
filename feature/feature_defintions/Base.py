from abc import ABC, abstractmethod


class FeatureBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_feature(self):
        pass
