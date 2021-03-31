#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import random
import glob
from config import mimic4, hidden_dim, lstm_layers, lr, dropout, dropout_w, dropout_conv, number_epochs, datasetpath

from mimic3models.models.DataLoader import LoadDataSets
from mimic3models.models.lstm_cnn import trainer, plotLoss

# Set random seeds
# https://pytorch.org/docs/stable/notes/randomness.html

def randseed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

randseed()


# Load Training, Test and Validation Data Sets
try:
    del train_data
except:
    pass

try:
    del test_data
except:
    pass

try:
    del val_data
except:
    pass

already_loaded = False

if mimic4:
    print("Training uses MIMIC-IV data.")
else:
    print("Training uses MIMIC-III data.")

dataloader_train, dataloader_val, dataloader_test = LoadDataSets(batch_size=64,mimic4=mimic4, datasetpath=datasetpath)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Train a specific LSTM+CNN model

# define threshold
threshold = 0.5
logit_threshold = torch.tensor (threshold / (1 - threshold)).log()
    
best_accuracy = 0
best_roc_auc = 0
best_loss = 100000

#hidden_dim, lstm_layers, lr, dropout, dropout_w, dropout_conv = (8, 2, 1e-3, 0.3, 0.2, 0.2)
print("Hyperparameters:")
print(f"hidden_dim = {hidden_dim}, lstm_layers = {lstm_layers}, lr = {lr}, dropout = {dropout}, dropout_w = {dropout_w}, dropout_conv = {dropout_conv}")

print()
print(f"Training is set for maximal number of {number_epochs} epochs.")

(best_loss, best_accuracy, best_roc_auc), train_loss, val_loss, modelsignature = trainer(dataloader_train, dataloader_val,
                                                                         number_epochs=number_epochs,
                                                                         hidden_dim=hidden_dim,
                                                                         lstm_layers=lstm_layers, lr=lr,
                                                                         dropout=dropout,
                                                                         dropout_w=dropout_w,
                                                                         dropout_conv=dropout_conv,
                                                                         best_loss=best_loss,
                                                                         best_accuracy=best_accuracy,
                                                                         best_roc_auc=best_roc_auc,
                                                                         early_stopping=0,
                                                                         verbatim=True)


print("Results on validation data set:")
print(f"Best loss={best_loss}, best accuracy={best_accuracy}, and best AUC={best_roc_auc}")

filename_loss = glob.glob(f"*{modelsignature}*loss-{best_loss}*.pth")[0]
filename_acc = glob.glob(f"*{modelsignature}*acc-{best_accuracy}*.pth")[0]
filename_auc = glob.glob(f"*{modelsignature}*auc-{best_roc_auc}.pth")[0]

print()
print("These are the statedict files with the best values when evaluated against the validation data set:")
print(f"Filename best loss = {filename_loss}")
print(f"Filename best accuracy = {filename_acc}")
print(f"Filename best AUC = {filename_auc}")

plotLoss(train_loss, val_loss)
print("Training of the model finished.")
