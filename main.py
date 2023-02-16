import argparse
import os
import os.path as osp
import random
import sys
import yaml

import torch
import numpy as np

import utils

from tqdm.notebook import tqdm
from utils import AverageMeter,get_lr
from data_utils import get_dataloader
from models import models
from train import train, validation
from utils import convert_dict_to_tuple
import val_on_test
import math
from torch import nn
def main():
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    with open("config/baseline_mcs.yml") as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    outdir = osp.join(config.outdir, config.exp_name)
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_loader, val_loader = get_dataloader.get_dataloaders(config)

    print("Loading model...")
    net = models.load_model(config)
    if config.num_gpu > 1:
        net = torch.nn.DataParallel(net)
    print("Done!")

    criterion, optimizer, scheduler = utils.get_training_parameters(config, 
                                                                    net)
    train_epoch = tqdm(range(config.train.n_epoch), dynamic_ncols=True, 
                       desc='Epochs', position=0)
    # main process
    
    

    


    for epoch in train_epoch:
        net.train()

        loss_stat = AverageMeter('Loss')
        acc_stat = AverageMeter('Acc.')

        train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

        for step, (img1, img2) in enumerate(train_iter):
            out1 = net.encode_image(img1.cuda().to(memory_format=torch.contiguous_format))
            out2 = net.encode_image(img2.cuda().to(memory_format=torch.contiguous_format))
            logits = out2 @ out1.T
            
            
            num_of_samples = img1.shape[0]
            loss =  criterion(logits, torch.arange(num_of_samples).cuda())
            acc=np.mean(np.array(torch.argmax(logits.to("cpu"), axis=1) == torch.arange(num_of_samples)))
            
            acc_stat.update(acc, num_of_samples)
            loss_item=loss.detach().cpu().item()
            loss_stat.update(loss_item, num_of_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

          
           

            acc_val, acc_avg = acc_stat()
            loss_val, loss_avg = loss_stat()
            train_iter.set_description('Epoch: {}; step: {}; loss: {:.4f}; acc: {:.4f}; acc_avg: {:.4f}; lr: {:.8f}'.format(epoch, step, loss_avg,acc,acc_avg,get_lr(optimizer)))

          

            if step % config.train.freq_vis == 0 and not step == 0:

                train_iter.set_description('Epoch: {}; step: {}; loss: {:.4f}; acc: {:.4f}; acc_avg: {:.4f}; lr: {:.5f}'.format(epoch, step, loss_avg,acc,acc_avg,get_lr(optimizer)))
            #scheduler.step()

            if step%500s==0:
                basel=val_on_test.MCS_BaseLine_Ranker(net,config.test.development_test_data,config.test.gallery_csv,config.test.query_csv)
                epoch_avg_acc=basel.predict_product_ranks()
                print(epoch_avg_acc)
# smooth the loss

        print('Train process of epoch: {} is done; \n loss: {:.4f}'.format(epoch, loss_avg))

        
        
        utils.save_checkpoint(net, optimizer, scheduler, epoch, outdir)
        print("model saved new epoch is starting...")
        
        
        
        #epoch_avg_acc = validation(net, val_loader, criterion, epoch)
"""
            if step==0:
                lr_find_loss.append(loss_item)
            else:
                loss = smoothing  * loss_item + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(loss_item)
            



outs = net.encode_image(imgs.cuda().to(memory_format=torch.contiguous_format))

logits = torch.matmul(outs , outs.T)
print(logits.shape)
num_of_samples = imgs.shape[0]
y=y.reshape(32,1)
mask=torch.eq(y, y.T)
mask = mask.float()
print(mask.shape)
loss =  criterion(logits, mask.to("cuda:0"))
acc=np.mean(np.array(torch.argmax(logits.to("cpu"), axis=1) == torch.arange(num_of_samples)))
"""
        
"""
if step%2000==0:
    basel=val_on_test.MCS_BaseLine_Ranker(net,"development_test_data/","development_test_data/gallery.csv","development_test_data/queries.csv")
    epoch_avg_acc=basel.predict_product_ranks()
    print(epoch_avg_acc)
"""

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to config file.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main()
