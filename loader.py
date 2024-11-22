import numpy as np
import sys, os, glob, re
from torch.utils.data import DataLoader, Dataset
import torch

class NPY2Dataset(Dataset):
    def __init__(self, data, labels, C, H, W):
        #self.data = data.unsqueeze(1)
        self.data = data.reshape(-1, C, H, W)

        #print("1",self.data.size())
        #print("1",self.data.type())
        self.labels = labels
        #print("2",self.labels.size())
        #print("2",self.labels.type())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class _NPY2Dataset(Dataset):
    def __init__(self, data, labels, C, H, W):
        # for single npy input file 
        #npt =torch.div(npt, npt.max())
        if split == "train" :
            self.data = data[:int(len(data) * valid_ratio)] 
            self.labels = labels[:int(len(labels) * valid_ratio)] 
        if split == "val" :
            self.data = data[int(len(data) * valid_ratio):]
            self.labels = labels[int(len(labels) * valid_ratio):]

        # assume 1 channel input data
        self.data = data.reshape(-1, C, H, W)

        print("1",self.data.size())
        print("1",self.data.type())
        print("2",self.labels.size())
        print("2",self.labels.type())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def npy_loader(path):
    return torch.from_numpy(np.load(path))

