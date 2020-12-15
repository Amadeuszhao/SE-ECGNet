# -*- coding: utf-8 -*-
import os


class Config:
    # path to your root dir
    root = r'/home'
    # path to training data dir
    train_dir = '/home/lizhe/zw/test/ecg/data/all_data'
    # path to test data dir
    test_dir = '/root/zw/test_/data/hf_round1_testA/testA'
    # path to train_label file
    train_label = os.path.join(root, 'hf_round2_train.txt')
    # path to test_label file
    test_label = os.path.join(root, 'hf_round1_subA.txt')
    # path to arrythmia file
    arrythmia = os.path.join(root, 'hf_round2_arrythmia.txt')
    train_data = os.path.join(root, 'train.pth')

    # for train
    # select train model SE_ECGNet ECGNet BiRCNN resnet34 
    model_name = 'resnet34'
    # learning rate decay method
    stage_epoch = [32,64,128,256]
    # batch_size
    batch_size = 64
    # number of labels
    num_classes = 34
    # max_epoch
    max_epoch = 256
    # resampling points(default 2048)
    target_point_num = 2048
    # model save dir
    ckpt = 'ckpt'
    # learning rate
    lr = 1e-3
    # saved current weight path
    current_w = 'current_w.pth'
    # saved best weight path
    best_w = 'best_w.pth'
    # learning rate decay lr/=lr_decay
    lr_decay = 10

    #for test
    temp_dir=os.path.join(root,'temp')


config = Config()
