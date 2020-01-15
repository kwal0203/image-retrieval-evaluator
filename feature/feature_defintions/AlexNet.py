from feature.feature_defintions.Base import FeatureBase
from sklearn.preprocessing import normalize
from torchvision import models, transforms
from PIL import Image
from torch import nn

import torch
import sys

# AlexNet model pre-trained on ImageNet dataset
# NOTE: Report feature taken from model.classifier[1] (layer: 2)
class AlexNet(FeatureBase):
    def __init__(self, config):
        super().__init__(config=config)
        self.transforms = self.transforms_create()
        self.model = self.model_create(config)

    # Comments...
    # Final layer must be called 'classifier.6.weight' in the incoming
    # state_dict
    # Feature can only be taken from layers of the classifier
    def model_create(self, config):
        feature_path = config['feature_path']
        feature_name = config['feature_name']
        layer = config['layer']

        # Load a model we trained or use ImageNet pretrained one otherwise
        model = models.alexnet(pretrained=True)
        if 'load' in feature_name:
            state_dict = torch.load(feature_path)
            num_features = model.classifier[6].in_features
            num_outputs = len(state_dict['classifier.6.weight'])
            model.classifier[6] = nn.Linear(num_features, num_outputs)
            model.load_state_dict(torch.load(feature_path))

        # Truncate network to level we want to get the feature from
        model.classifier = model.classifier[:layer]
        for param in model.parameters():
            param.requires_grad = False

        # Send model to GPU and set to evaluation mode
        model = model.cuda().eval()
        return model

    # Comments...
    def transforms_create(self):
        _transforms = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
        ]
        return transforms.Compose(_transforms)

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
        result = normalize(result).flatten()
        assert(result.size == 4096)
        return result

