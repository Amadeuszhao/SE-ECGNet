# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 19:47

@ author: javis
'''
import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal


def resample(sig, target_point_num=None):
    '''
    resample the original signal
    :param sig: original signal
    :param target_point_num
    :return: resampled signal
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def transform(sig, train=False):
    '''
    resample the original signal
    :param sig: original signal
    :return: resampled signal in tensor form
    '''
    sig = resample(sig, config.target_point_num)
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(config.train_data)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(config.train_dir, fid)
        df = pd.read_csv(file_path, sep=' ').values
        x = transform(df, self.train)
        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    x, target = ECGDataset(config.train_data)[0]
    print(x.shape)
    print(target.shape)
