import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision
from sklearn.model_selection import train_test_split
import torch.distributed as dist
import torch.nn.functional as F


class _SplitTestDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitTestDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transform = None
    def __getitem__(self, key):
        x,y = self.underlying_dataset[self.keys[key]]
        if self.transform is not None:
            x = self.transform(x)
        return x,y
    def __len__(self):
        return len(self.keys)
    
class _SplitTrainDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitTrainDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transform = None
    def __getitem__(self, key):
        x,y = self.underlying_dataset[self.keys[key]]
        if self.transform is not None:
            img_w = self.transform[0](x)
            img_s = self.transform[1](x)

        return img_w, img_s, y
    def __len__(self):
        return len(self.keys)

class _SplitTrainDatasetNew(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitTrainDatasetNew, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transform = None
    def __getitem__(self, key):
        x,y = self.underlying_dataset[self.keys[key]]
        if self.transform is not None:
            img_w = self.transform[0](x)
            img_s = self.transform[1](x)
            img_s1 = self.transform[1](x)

        return img_w, img_s, img_s1, y
    def __len__(self):
        return len(self.keys)
    
class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transform = None
    def __getitem__(self, key):
        x,y = self.underlying_dataset[self.keys[key]]
        if self.transform is not None:
            x = self.transform(x)
        return x,y
    def __len__(self):
        return len(self.keys)

def few_shot_subset(targets,n_shot):
    '''
    targets : torch.tensor contains labels of each data points
    n_shot : number of data point for each class
    Returns list contains indices of n_shot dataset
    '''
    # non repeat
    class_, counts = torch.unique(targets,return_counts=True)
    indices = []
    for i, count in enumerate(counts):
        # sample in each class
        idx = torch.where(targets==class_[i])[0]
        # small class->not sample
        if count < n_shot+1:
            raise ValueError(f"Class {class_[i]} only have {count} samples, {n_shot}-Shot is not available")
        else:
            # random sample
            temp = torch.randperm(len(idx))
            # train->n_shot
            trn_idx = idx[temp[:n_shot]]
        # tolistt
        indices.extend(trn_idx.tolist())

    return indices

def split_dataset(dataset, seed=0, ratio=0.2):
    assert(ratio <= 0.5)
    keys = list(range(len(dataset)))
    classes = dataset.targets
    keys_1,keys_2 = train_test_split(keys,test_size=ratio,random_state=seed,stratify=classes)

    return keys_1, keys_2

    #keys = [i for i in range(len(dataset))]
    #SEED = seed
    #torch.manual_seed(SEED)
    #torch.cuda.manual_seed(SEED)
    #np.random.seed(SEED)
    #random.seed(SEED)
    #random.shuffle(keys)
    #return keys

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def soft_criterion(s_pre, t_pre):
    batch_size, dim = s_pre.size()
    return -torch.sum(torch.mul(t_pre, torch.log(s_pre + 1e-4))) / (batch_size )

def cdd(output_t1, output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss 
