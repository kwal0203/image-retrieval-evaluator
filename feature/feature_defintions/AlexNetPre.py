from feature.feature_defintions.Base import FeatureBase
from torchvision import models, transforms

# AlexNet model pre-trained on ImageNet dataset
class AlexNetPre(FeatureBase):
    def __init__(self, config):
        super().__init__(config=config)
        self.device = torch.device("cuda:0")
        self.model = models.alexnet(pretrained=True).to(self.device)
        self.transform = transforms.Compose([
                             transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]
                             )
                         ])
        self.output_vector = None
        self.model.eval()

    def get_activations(self, layer, i, o):
        self.output_vector.copy_(o.data.squeeze())

    def get_feature(self, image):
        # Transform input image same as AlexNet training
        _image = self.transform(image).unsqueeze(0)
        _image = _image.to(self.device)

        # Attach hook to network and capture activations on forward pass
        fc_6 = self.model.classifier[1]
        hook = fc_6.register_forward_hook(self.get_activations)
        self.model(_image)
        hook.remove()

        # Return normalized 4096 dimension activation vector (FC6)
        return preprocessing.normalize([self.output_vector.numpy()])[0]