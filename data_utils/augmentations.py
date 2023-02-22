import torchvision as tv
import open_clip
from torchvision.transforms import transforms
from RandAugment import RandAugment
def _convert_to_rgb(image):
    return image.convert('RGB')
def get_train_aug(config):
    if config.dataset.augmentations == 'default':
        train_augs = tv.transforms.Compose([
            tv.transforms.Resize(size=256,interpolation=tv.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            tv.transforms.RandomCrop((224,224)),
            tv.transforms.ColorJitter(),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomRotation(degrees=(0, 180)),
            _convert_to_rgb,
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    if config.dataset.augmentations == 'augmix':
        train_augs = tv.transforms.Compose([
            tv.transforms.RandAugment(2,14),
            tv.transforms.Resize(size=256,interpolation=tv.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            tv.transforms.CenterCrop((224,224)),
            _convert_to_rgb,
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        #train_augs.transforms.insert(1, RandAugment(15, 0.5))
    else:
        raise Exception("Unknonw type of augs: {}".format(
            config.dataset.augmentations
        ))
    return train_augs


def get_val_aug(config):
    if config.dataset.augmentations_valid == 'default':
        
        _, _, val_augs = open_clip.create_model_and_transforms(config.model_name, pretrained=config.model_dataset_name)
    else:
        raise Exception("Unknonw type of augs: {}".format(
            config.dataset.augmentations
        ))
    return val_augs

