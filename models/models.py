from torch import nn
from torchvision import models
import torch
from collections import namedtuple, OrderedDict
import open_clip
def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    model, _, _ = open_clip.create_model_and_transforms(config.model_name, pretrained=config.model_dataset_name)
    model.to("cuda:0")
    return model.visual
"""
arch = config.model.arch
num_classes = config.dataset.num_of_classes
if arch.startswith('resnet'):
    model = models.__dict__[arch](weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
else:
    raise Exception('model type is not supported:', arch)
checkpoint = torch.load("experiments/baseline_mcs/baseline_model.pth",
                            map_location='cuda')['state_dict']
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to('cuda')
"""
