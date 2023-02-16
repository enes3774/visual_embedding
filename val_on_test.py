import numpy as np

import yaml

import numpy as np
import torch
import torchvision.models as models

from collections import OrderedDict

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pandas
import mean_average_precision
import sys
sys.path.append('./MCS2023_baseline/')

from data_utils.augmentations import get_val_aug
from data_utils.dataset import SubmissionDataset
from utils import convert_dict_to_tuple
import open_clip
class MCS_BaseLine_Ranker:
    def __init__(self, model,dataset_path, gallery_csv_path, queries_csv_path):
        """
        Initialize your model here
        Inputs:
            dataset_path
            gallery_csv_path
            queries_csv_path
        """
        # Try not to change
        self.dataset_path = dataset_path
        self.gallery_csv_path = gallery_csv_path
        self.queries_csv_path = queries_csv_path
        self.max_predictions = 1000

        # Add your code below

        checkpoint_path = 'experiments/baseline_mcs/model_0002.pth'
        self.batch_size = 16

        self.exp_cfg = 'config/baseline_mcs.yml'
        self.inference_cfg = 'config/inference_config.yml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(self.exp_cfg) as f:
            data = yaml.safe_load(f)
        self.exp_cfg = convert_dict_to_tuple(data)

        with open(self.inference_cfg) as f:
            data = yaml.safe_load(f)
        self.inference_cfg = convert_dict_to_tuple(data)

        print('Creating model and loading checkpoint')
        self.model=model
        self.model.to(self.device)
        print('Weights are loaded, fc layer is deleted!')
        self.embedding_shape=512

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def predict_product_ranks(self):
        """
        This function should return a numpy array of shape `(num_queries, 1000)`. 
        For ach query image your model will need to predict 
        a set of 1000 unique gallery indexes, in order of best match first.

        Outputs:
            class_ranks - A 2D numpy array where the axes correspond to:
                          axis 0 - Batch size
                          axis 1 - An ordered rank list of matched image indexes, most confident prediction first
                            - maximum length of this should be 1000
                            - predictions above this limit will be dropped
                            - duplicates will be dropped such that the lowest index entry is preserved
        """

        gallery_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.gallery_csv_path,
            transforms=get_val_aug(self.exp_cfg)
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=0
        )

        query_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.queries_csv_path,
            transforms=get_val_aug(self.exp_cfg), with_bbox=True
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=0
        )

        print('Calculating embeddings')
        gallery_embeddings = np.zeros((len(gallery_dataset), self.embedding_shape))
        query_embeddings = np.zeros((len(query_dataset), self.embedding_shape))

        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                images = images.to(self.device)
                outputs = self.model.encode_image(images)
                outputs = outputs.data.cpu().numpy()
                gallery_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
            
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                images = images.to(self.device)
                outputs = self.model.encode_image(images)
                outputs = outputs.data.cpu().numpy()
                query_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
        
        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)
        distances = pairwise_distances(query_embeddings, gallery_embeddings,metric="braycurtis")
        sorted_distances = np.argsort(distances, axis=1)[:, :1000]
        quer=pandas.read_csv(self.queries_csv_path)['product_id'].values
        gal=pandas.read_csv(self.gallery_csv_path)['product_id'].values
        result=mean_average_precision.calculate_map(sorted_distances, quer, gal)
        print(result)
        return result
#braycurtis