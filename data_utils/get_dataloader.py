import torch

from data_utils import dataset, augmentations
import random
import pandas
from torch.utils.data import DataLoader, Dataset, Sampler
from itertools import combinations
"""
        data_by_class = {}
        grouped = dataset.groupby('class')
        for name, group in grouped:
            data_by_class[name] = group['name'].tolist()
        data_list = []
        for class_data in data_by_class.values():
            data_list.append(class_data)
        self.data_list=data_list
"""
"""
class DifferentClassSampler(Sampler):
    def __init__(self, dataset,batch_size):
        self.batch_size=batch_size
        self.data_by_class = {}
        grouped = dataset.groupby('class')
        for name, group in grouped:
            self.data_by_class[name] = group['name'].tolist()
        perm_data={}
        for i in range(len(self.data_by_class)):
            classs=self.data_by_class[i]
            perm_data[i]=list(combinations(classs, 2))
        batch_indices=[]
        while len(perm_data)>48:
            batch=[]
            
            samples = random.sample(sorted(perm_data), self.batch_size)
            for sample in samples:
                el=random.randint(0,len(perm_data[sample])-1)
                batch.append(perm_data[sample][el])
                perm_data[sample].pop(el)
                if len(perm_data[sample])==0:
                    del perm_data[sample]
            batch_indices.append(batch)
        self.iter_batch=[item for sublist in batch_indices for item in sublist]
    def __iter__(self):
        
        return iter(self.iter_batch)
    def __len__(self):
        return len(self.iter_batch)
"""

class DifferentClassSampler(Sampler):
    def __init__(self, dataset,batch_size):
        self.batch_size=batch_size
        self.data_by_class = {}
        grouped = dataset.groupby('class')
        for name, group in grouped:
            self.data_by_class[name] = group['name'].tolist()
        perm_data={}
        for i in range(len(self.data_by_class)):
            classs=self.data_by_class[i]
            perm_data[i]=list(combinations(classs, 2))
        batch_indices=[]
        while len(perm_data)>self.batch_size:
            batch=[]
            
            samples = random.sample(sorted(perm_data), self.batch_size)
            for sample in samples:
                el=random.randint(0,len(perm_data[sample])-1)
                batch.append(perm_data[sample][el])
                perm_data[sample].pop(el)
                if len(perm_data[sample])==0:
                    del perm_data[sample]
            batch_indices.append(batch)
        self.iter_batch=[item for sublist in batch_indices for item in sublist]
    def __iter__(self):
        
        return iter(self.iter_batch)
    def __len__(self):
        return len(self.iter_batch)
import copy
class DifferentClassSampler_random(Sampler):
    def __init__(self, dataset,batch_size):
        self.batch_size=batch_size
        self.data_by_class = {}
        grouped = dataset.groupby('class')
        for name, group in grouped:
            self.data_by_class[name] = group['name'].tolist()
        perm_data={}
        for i in range(len(self.data_by_class)):
            classs=self.data_by_class[i]
            perm_data[i]=list(combinations(classs, 2))
        batch_indices=[]
        perm_data_copy=copy.deepcopy(perm_data)
        while len(batch_indices)<10000:
            batch=[]
            
            samples = random.sample(sorted(perm_data), self.batch_size)
            for sample in samples:
                el=random.randint(0,
                                  len(perm_data[sample])-1)
                batch.append(perm_data[sample][el])
                perm_data[sample].pop(el)
                if len(perm_data[sample])<1:
                    perm_data[sample]=copy.deepcopy(perm_data_copy[sample])
            batch_indices.append(batch)
        self.iter_batch=[item for sublist in batch_indices for item in sublist]
        print(self.iter_batch[:100])
    def __iter__(self):
        
        return iter(self.iter_batch)
    def __len__(self):
        return len(self.iter_batch)
def get_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")
    sampler = DifferentClassSampler_random(pandas.read_csv(config.dataset.train_list),config.dataset.batch_size)
    train_dataset = dataset.Product10KDataset(
        path=config.dataset.train_prefix,
        root=config.dataset.train_prefix, annotation_file=config.dataset.train_list,
        transforms=augmentations.get_train_aug(config)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=True,#!!!! hızlandırıyor
        drop_last=True,
        sampler=sampler
    )
    print("Done.")

    print("Preparing valid reader...")
    val_dataset = dataset.Product10KDataset(
        path=config.dataset.val_prefix,
        root=config.dataset.val_prefix, annotation_file=config.dataset.val_list,
        transforms=augmentations.get_val_aug(config)
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True,
        sampler=sampler
    )
    print("Done.")
    return train_loader, valid_loader
