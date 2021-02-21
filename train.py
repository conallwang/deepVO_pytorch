import os
import argparse
import torch
import math
import numpy as np

from tqdm import tqdm
from time import time

from model import DeepVO
from dataset import DeepVODataset
from config import *
from utils.misc import display_loss_tb, pre_create_file_train, to_var

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# y and label is [batch_size, 6]
def loss_func(y, label):
    return torch.mean(kapa * torch.pow((y[:, :3] - label[:, :3]), 2) + torch.pow((y[:, 3:] - label[:, 3:]), 2))

def run_batch(sample, model, loss_func=None, optimizer=None):
    model.train()

    count = 0
    loss_mean = 0
    for sample_batch in sample:
        img_1 = to_var(sample_batch['img_1'])
        img_2 = to_var(sample_batch['img_2'])
        label_pre = model(img_1, img_2)

        loss = loss_func(label_pre, sample_batch['label'].reshape(-1, 6).to(device))
        loss_mean += loss.item()
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss_mean /= count
    return loss_mean

def run_test():
    pass

def train(args):
    dir_model, dir_log = pre_create_file_train(model_path, log_path, args)
    writer = SummaryWriter(dir_log)
    model = DeepVO().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)
    pbar = tqdm(range(args.epoch_max))

    data_set_t_1 = DeepVODataset(seq=2, interval=1, phase='train')
    data_set_t_2 = DeepVODataset(seq=4, interval=2, phase='train')
    data_set_t_3 = DeepVODataset(seq=8, interval=4, phase='train')
    # data_set_v = DeepVODataset(seq=4, interval=40, phase='valid')
    loader_t_1 = DataLoader(data_set_t_1, batch_size=16, shuffle=True)
    loader_t_2 = DataLoader(data_set_t_2, batch_size=8, shuffle=True)
    loader_t_3 = DataLoader(data_set_t_3, batch_size=4, shuffle=True)
    # loader_v = DataLoader(data_set_v, batch_size=4, shuffle=False)

    step_per_epoch = int(math.ceil(data_set_t_1.__len__()/loader_t_1.batch_size))
    step_val = int(math.floor(step_per_epoch / 3))      # vaildate 3 times one epoch

    for epoch in pbar:

        # test
        if (epoch + 1) % args.epoch_test == 0:
            run_test()

        loss_list = []
        for step, (sample_t_1, sample_t_2, sample_t_3) in enumerate(zip(loader_t_1, loader_t_2, loader_t_3)):
            tic = time()
            step_global = epoch * step_per_epoch + step
            loss = run_batch(sample=[sample_t_1, sample_t_2, sample_t_3], model=model, loss_func=loss_func, optimizer=optimizer)
            loss_list.append(loss)
            
            hour_per_epoch = step_per_epoch * ((time() - tic) / 3600)

            if (step + 1) % 5 == 0:
                display_loss_tb(hour_per_epoch, pbar, step, step_per_epoch, optimizer, loss, writer, step_global)
        
        # save model
        if args.save and (epoch + 1) % args.epoch_save == 0:
            print(f'\nSave model: {dir_model}/model-{epoch+1}.pkl')
            torch.save(model.state_dict(), f'{dir_model}/model-{epoch+1}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr_base', type=float, default=1e-4, help='initial learning rate')

    parser.add_argument('--save_model', action='store_true')
    
    parser.add_argument('--epoch_max', type=int, default=100, help='max iteration')
    parser.add_argument('--epoch_test', type=int, default=10, help='how many epoch to test a complete sequence')
    parser.add_argument('--epoch_save', type=int, default=10, help='how many epoch to save models')

    parser.add_argument('--net_name', type=str, default='cnn-lstm', help='cnn-lstm(default)')
    parser.add_argument('--dir0', type=str, default='default', help='using date to identify files (default is default)')

    args = parser.parse_args()
    
    train(args)