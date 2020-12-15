# -*- coding: utf-8 -*-
import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
#from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ECGDataset
from config import config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)


# save current weight and update new best weight
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter,acc_meter,recall_meter,precision_meter,loss_meter, it_count = 0, 0, 0,0,0,0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1 = utils.calc_f1(target, output, threshold)
            acc = utils.cal_accuracy_score(target,output)
            recall = utils.cal_recall_score(target,output)
            precision = utils.cal_percision_score(target,output)
            f1_meter += f1
            acc_meter +=acc
            recall_meter +=recall
            precision_meter += precision
    return loss_meter / it_count, f1_meter / it_count , acc_meter / it_count , recall_meter / it_count, precision_meter / it_count


def train(args):
    # get model 
    model = getattr(models, config.model_name)()
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    print(model)
    # data
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = utils.FocalLoss()
    # model save dir
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    utils.mkdirs(model_save_dir)
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # train from last save point
    if args.resume:
        if os.path.exists(args.ckpt):  # weight path
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    #logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>start training<=========
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        # logger.log_value('train_loss', train_loss, step=epoch)
        # logger.log_value('train_f1', train_f1, step=epoch)
        # logger.log_value('val_loss', val_loss, step=epoch)
        # logger.log_value('val_f1', val_f1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage}
        save_ckpt(state, best_f1 < val_f1, model_save_dir)
        best_f1 = max(best_f1, val_f1)
        print(best_f1)
        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

# test 
def val(args):
    list_threhold = [0.5]
    model = models.resnet34()
    if args.ckpt: model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    for threshold in list_threhold:
        val_loss, val_f1,val_acc,val_recall,val_precision = val_epoch(model, criterion, val_dataloader, threshold)
        print('threshold %.2f val_loss:%0.6e val_f1:%.8f\n , val_acc: %.8f\n val_recall: %.8f\n val_precision : %.8f\n' % (threshold, val_loss, val_f1,val_acc,val_recall,val_precision))



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    if (args.command == "test"):
        test(args)
    if (args.command == "val"):
        val(args)
