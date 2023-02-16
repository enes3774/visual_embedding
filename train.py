import numpy as np
import torch

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from utils import AverageMeter,get_lr
import val_on_test
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          config, epoch) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch (int): epoch number
    :return: None
    """
    model.train()

    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')

    train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

    for step, (img1, img2) in enumerate(train_iter):
        out1 = model.encode_image(img1.cuda().to(memory_format=torch.contiguous_format))
        out2 = model.encode_image(img2.cuda().to(memory_format=torch.contiguous_format))
        logits = out2 @ out1.T
        num_of_samples = img1.shape[0]
        loss =  criterion(logits, torch.arange(num_of_samples).to("cuda:0"))
        acc=np.mean(np.array(torch.argmax(logits.to("cpu"), axis=1) == torch.arange(num_of_samples)))
        acc_stat.update(acc, num_of_samples)

        loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      
       

        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        train_iter.set_description('Epoch: {}; step: {}; loss: {:.4f}; acc: {:.4f}; acc_avg: {:.4f}; lr: {:.5f}'.format(epoch, step, loss_avg,acc,acc_avg,get_lr(optimizer)))

      

        if step % config.train.freq_vis == 0 and not step == 0:

            train_iter.set_description('Epoch: {}; step: {}; loss: {:.4f}; acc: {:.4f}; acc_avg: {:.4f}; lr: {:.5f}'.format(epoch, step, loss_avg,acc,acc_avg,get_lr(optimizer)))

    print('Train process of epoch: {} is done; \n loss: {:.4f}'.format(epoch, loss_avg))


def validation(model: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               epoch) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: float: avg acc
     """
    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')

    with torch.no_grad():
        model.eval()
        val_iter = tqdm(val_loader, desc='Val', dynamic_ncols=True, position=2)

        for step, (x, y) in enumerate(val_iter):
            out = model(x.cuda().to(memory_format=torch.contiguous_format))
            loss = criterion(out, y.cuda())
            num_of_samples = x.shape[0]

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)

            scores = torch.softmax(out, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            acc = np.mean(gt == predict)
            acc_stat.update(acc, num_of_samples)
            val_iter.set_description('acc: {:.2f};'.format(acc))

        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        print('Validation of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}'.format(epoch, loss_avg, acc_avg))
        return acc_avg
