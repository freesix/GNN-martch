import numpy as np
import torch
import torch.utils.data as data
import cv2
import os 
import h5py
import random

import sys 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

from utils import evaluation_utils, train_utils

torch.multiprocessing.set_sharing_strategy('file_system') #设置多进程共享内存的策略,策略选择跟操作系统和硬件有关


class Offline_Dataset(data.Dataset):
    def __init__(self, config, mode):
        assert mode=='train' or 'valid'

        self.config = config 
        self.mode =mode 
        metadir=os.path.join(config.dataset_path, 'valid') if mode=='valid' else os.path.join(config.dataset_path, 'train') #原始路径
        #用于读取pair_num.txt文件内容
        pair_num_list=np.loadtxt(os.path.join(metadir, 'pair_num.txt'), dtype=str)
        self.total_pairs=int(pair_num_list[0,1])
        self.pair_seq_list, self.accu_pair_num=train_utils.parse_pair_seq(pair_num_list)

    def collate_fn(self, batch):
        