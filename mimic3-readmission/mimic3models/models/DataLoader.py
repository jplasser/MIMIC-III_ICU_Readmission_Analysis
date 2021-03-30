import pandas as pd
import pickle
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import random

import mimic3models.metrics as m
import matplotlib.pyplot as plt

import hiplot as hip
import re

# Load test and train data from Pickle files

class MIMICDataset(Dataset):
    """MIMIC dataset."""

    def __init__(self, data):
        """
        Args:
            data tuple(numpy.ndarray, list): data structured as tuple containing x which is a numpy array and y that is a list of values
        """
        self.x = data[0]
        self.y = data[1]
        #self.index = (torch.arange(48)-24).float()/48

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #x = torch.cat((self.index[:, None], torch.tensor(self.x[idx], dtype=torch.float32)), 1)
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return [x, y]

def LoadDataSets(batch_size=64, mimic4=False):
    # read train and test data for MIMIC-III

    #mimic4 = False
    if mimic4:
        data_path = "../readmission/train_data_mimic4/"
    else:
        data_path = "../readmission/train_data/"

    #already_loaded = True
    #try:
    #    train_data
    #except NameError as e:
    #    already_loaded = False

    #if not already_loaded:
    print(f"Loading train, test and validation data... from {data_path}")
    train_data = pickle.load(open(f"{data_path}train_data", "rb" ))
    val_data = pickle.load(open(f"{data_path}val_data", "rb" ))
    test_data = pickle.load(open(f"{data_path}test_data", "rb" ))
    
    print("Dimensions Train Data: ",len(train_data[0]), len(train_data[0][0]), len(train_data[0][0][0]))
    print("Dimensions: ",len(val_data[0]), len(val_data[0][0]), len(val_data[0][0][0]))
    print("Dimensions: ",len(test_data['data'][0]), len(test_data['data'][0][0]), len(test_data['data'][0][0][0]))
    
    ds_train = MIMICDataset(train_data)
    ds_val = MIMICDataset(val_data)
    ds_test = MIMICDataset(test_data['data'])
    
    # default batch size
    #batch_size = 64
    num_workers = 1

    # helper for random seed
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader_train = DataLoader(ds_train, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker)

    dataloader_val = DataLoader(ds_val, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)

    dataloader_test = DataLoader(ds_test, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)
    
    return dataloader_train, dataloader_val, dataloader_test
