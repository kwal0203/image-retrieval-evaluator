from feature.feature_defintions.Base import FeatureBase
from sklearn.preprocessing import normalize
from torchvision import models, transforms
from PIL import Image
from torch import nn

import torch


# User defined model
class MyModel(FeatureBase):
    def __init__(self, config):
        super().__init__(config=config)
        self.transforms = self.transforms_create()
        self.model = self.model_create(config)

    # Comments...
    def model_create(self, config):
        feature_path = config['feature_path']
        feature_name = config['feature_name']
        layer = config['layer']

        # DEFINE MODEL HERE
        model = "'feature/feature_definitions/MyModel.py'"
        assert(False), "You need to define your model dude in {}".format(model)

        # Send model to GPU and set to evaluation mode
        model = model.cuda().eval()
        return model

    # Comments...
    def transforms_create(self):
        _transforms = [
                # DEFINE TRANSFORMS HERE
        ]
        return transforms.Compose(_transforms)

    # Comments...
    def image_convert(self, image):
        image = Image.fromarray(image)
        image = self.transforms(image).unsqueeze(0)

        # DEFINE CUSTOM IMAGE OPERATIONS HERE

        image = image.cuda()
        return image

    # Comments...
    def get_feature(self, image):
        image = self.image_convert(image)
        result = self.model(image).cpu()
        result = normalize(result).flatten()

        # ADD SIZE OF FEATURE HERE
        assert(result.size == 4096)
        return result

