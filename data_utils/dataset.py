import os

import cv2
import pandas as pd
import torch.utils.data as data

from PIL import Image

def read_image(image_file):
    img = cv2.imread(
        image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


class Product10KDataset(data.Dataset):
    def __init__(self, path,root, annotation_file, transforms, is_inference=False,
                 with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.path=path

    def __getitem__(self, index):
        
        cv2.setNumThreads(6)
        img1,img2=index

       
        img1 = read_image(os.path.join(self.path,img1))
        img1 = Image.fromarray(img1)
        img2 = read_image(os.path.join(self.path,img2))
        img2 = Image.fromarray(img2)
        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        
        return img1,img2

    def __len__(self):
        return len(self.imlist)


#iki dataset için de submission için de transformları değiştir.
# Augmenttion teknikleri uygula düşük rl kullan ve her deneemede modelin veridği sonucu gözlmele
#mesela bazı layerları dondur dene gibi
class SubmissionDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imlist)
