from feature.feature_defintions.Base import FeatureBase
from sklearn.preprocessing import normalize
from torchvision import models, transforms
from PIL import Image

import torch


# AlexNet model pre-trained on ImageNet dataset
class AlexNetPre(FeatureBase):
    def __init__(self, config):
        super().__init__(config=config)
        self.transforms = self.transforms_create()
        self.model = self.model_create(config['layer'])

    # Comments...
    def model_create(self, layer):
        model = models.alexnet(pretrained=True)
        model.classifier = model.classifier[:layer]
        for param in model.parameters():
            param.requires_grad = False
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

