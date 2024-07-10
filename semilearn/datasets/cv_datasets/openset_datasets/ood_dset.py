# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torch.utils.data import Dataset
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_osssl_data, reassign_target


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['svhn'] = [0.4380, 0.4440, 0.4730]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['svhn'] = [0.1751, 0.1771, 0.1744]


def get_ood_dset(args, alg, name, num_classes, data_dir='./data'):
    name = name.split('_')[0]  # cifar10_openset -> cifar10
    if name == "cifar10":
        ood_set = ["svhn", "lsun", "gaussian", "uniform"]
    elif name == "cifar100":
        ood_set = ["svhn", "lsun", "gaussian", "uniform"]
    else:
        ood_set = ["svhn", "lsun", "gaussian", "uniform"]
    
    ood_dset = {}
    for dataset in ood_set:
        cur_data_dir = os.path.join(data_dir, dataset.lower())
        dset = eval("get_" + dataset)(cur_data_dir, alg, num_classes, args.img_size, 10000)
        ood_dset[dataset] = dset
    
    return ood_dset

def get_svhn(data_dir, alg, num_classes, img_size, len_per_dset=-1):
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['svhn'], std['svhn'])
    ])
    
    dset = torchvision.datasets.SVHN(data_dir, split="test", download=False)
    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
    total_len = data.shape[0]
    
    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]
        
    dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)
    return dset

def get_lsun(data_dir, alg, num_classes, img_size, len_per_dset=-1):
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    data = np.load(os.path.join(data_dir, 'LSUN_resize.npy'))
    targets = np.zeros(data.shape[0], dtype=int)
    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]
        
    dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)
    return dset

def get_gaussian(data_dir, alg, num_classes, img_size, len_per_dset=-1):
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    data = np.load(os.path.join(data_dir, 'Gaussian.npy'))
    targets = np.zeros(data.shape[0], dtype=int)
    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]
        
    dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)
    return dset

def get_uniform(data_dir, alg, num_classes, img_size, len_per_dset=-1):
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    data = np.load(os.path.join(data_dir, 'Uniform.npy'))
    targets = np.zeros(data.shape[0], dtype=int)
    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]
        
    dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)
    return dset