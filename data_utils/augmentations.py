import torchvision as tv
import open_clip
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
    else:
        raise Exception("Unknonw type of augs: {}".format(
            config.dataset.augmentations
        ))
    return train_augs


def get_val_aug(config):
    if config.dataset.augmentations_valid == 'default':
        
        _, _, val_augs = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31')
    else:
        raise Exception("Unknonw type of augs: {}".format(
            config.dataset.augmentations
        ))
    return val_augs

