import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

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

def LoadDataSets(batch_size=64, mimic4=False, datasetpath='../readmission/'):
    # read train and test data for MIMIC-III

    #mimic4 = False
    if mimic4:
        data_path = f"{datasetpath}train_data_mimic4/"
    else:
        data_path = f"{datasetpath}train_data/"

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
