# -*- coding: utf-8 -*-

import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from torch import nn


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cal_percision_score(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return precision_score(y_true, y_pre)

def cal_recall_score(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return recall_score(y_true, y_pre)



def cal_accuracy_score(y_true, y_pre,threshold=0.5):
    # y_true = y_true.cpu().view(-1)
    # y_pre =y_pre.cpu().view(-1,5)
    # print(y_pre.shape)
    # print(y_true.shape)
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return accuracy_score(y_pre,y_true)
# calcute score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)


def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# adjust learning rate 
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Multi label loss
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()

#focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true):
        y_pred = self.sigmoid(y_pred)#之前有sigmoid的话记得注释掉这一句
        fl = - self.alpha * y_true * torch.log(y_pred) * ((1.0 - y_pred) ** self.gamma) - (1.0 - self.alpha) * (1.0 - y_true) * torch.log(1.0 - y_pred) * (y_pred ** self.gamma)
        fl_sum = fl.sum()
        return fl_sum
