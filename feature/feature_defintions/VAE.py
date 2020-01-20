from feature.feature_defintions.Base import FeatureBase
from sklearn.preprocessing import normalize
from torchvision import models, transforms
from PIL import Image
from torch import nn

from skimage.color import rgb2gray

import torch
import sys


# NOTE: Report feature taken from xxx (layer: xxx)
# TODO: Add image dimension fields to the class (for use in resize transform)
class VAE(FeatureBase):
    def __init__(self, config):
        super().__init__(config=config)
        self.transforms = self.transforms_create()
        self.model = self.model_create(config)

    # Comments...
    # USE LAYER == 1 in feature_params.json to get 2048 feature
    def model_create(self, config):
        feature_path = config['feature_path']
        feature_name = config['feature_name']
        layer = config['layer']

        print(feature_path)

        # Define structure of network (identical to the one we trained)
        model = models.resnet50(pretrained=False)

        # Load the saved weights into the model we just defined
        state_dict = torch.load(feature_path)

        # The names in the state dict for the model I trained had 'encoder' at
        # the beginning so gotta remove it for load_state_dict() to work
        # correctly
        _tmp = 'encoder.'
        clip = len(_tmp)
        new_dict = {
                k[clip:]: v for k, v in state_dict.items() if k.startswith(_tmp
        )}
        model.load_state_dict(new_dict)

        for param in model.parameters():
            param.requires_grad = False

        model = nn.Sequential(
            *list(model.children())[:-layer]
        )
        # Send model to GPU and set to evaluation mode
        model = model.cuda().eval()
        return model

    # Comments...
    def transforms_create(self):
        my_transforms = [
                transforms.ToTensor(),
        ]
        return transforms.Compose(my_transforms)

    # Comments...
    def image_convert(self, image):
        image = Image.fromarray(image)
        image = self.transforms(image).unsqueeze(0)
        image = image.cuda()
        return image

    # Comments...
    def get_feature(self, image):
        image = self.image_convert(image)
        result = self.model(image).cpu()
        result = result.squeeze().reshape(1, -1)
        result = normalize(result).flatten()
        assert(result.size == 2048)
        return result

