#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from tqdm.auto import tqdm
import numpy as np
import random

import mimic3models.metrics as m
import matplotlib.pyplot as plt
import glob

from DataLoader import LoadDataSets
from lstm_cnn import LSTM_CNN4
from lstm_cnn import trainer, evaluate, calcMetrics, plotLoss, plotAUC


# # Set random seeds

# In[ ]:


#CUDA RNN and LSTM
#In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior. See torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.
# https://pytorch.org/docs/stable/notes/randomness.html

def randseed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

randseed()


# # Load Training, Test and Validation Data Sets
# 
# Set `mimic4` to `True` if you want to evaluate against MIMIC-IV, or to `False` for MIMIC-III.

# In[ ]:


# if you want to evaluate models with MIMIC-III then sey mimic=False
# if you want to evaluate models with MIMIC-IV then sey mimic=True
mimic4 = True


# In[ ]:


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

dataloader_train, dataloader_val, dataloader_test = LoadDataSets(batch_size=64,mimic4=mimic4)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# # Train a specific LSTM+CNN model

# In[ ]:


randseed()

# define threshold
threshold = 0.5
logit_threshold = torch.tensor (threshold / (1 - threshold)).log()


# # Run a sequence of trainings with one model to capture stats
# 
# Before running the cell set the best model hyperparameters for MIMIC-III/IV:
# 
# `hidden_dim, lstm_layers, lr, dropout, dropout_w, dropout_conv = (16, 2, 1e-3, 0.5, 0.3, 0.5)`
# 

# In[ ]:


number_epochs = 30
number_iterations = 30

overall_best_loss = 100000
overall_best_accuracy = 0.
overall_best_roc_auc = 0.
best_models = {}

# set the hyperparameters for the best model
hidden_dim, lstm_layers, lr, dropout, dropout_w, dropout_conv = (8, 2, 1e-3, 0.3, 0.2, 0.2)

results = []

for i in tqdm(range(number_iterations)):
    best_accuracy = 0
    best_roc_auc = 0
    best_loss = 100000
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
         early_stopping=5,
         verbatim=False)
    
    results.append([best_loss, best_accuracy, best_roc_auc])

    if best_loss < overall_best_loss:
        best_models['loss'] = (best_loss, best_accuracy, best_roc_auc), train_loss, val_loss, modelsignature
        overall_best_loss = best_loss 
        
    if best_accuracy > overall_best_accuracy:
        best_models['acc'] = (best_loss, best_accuracy, best_roc_auc), train_loss, val_loss, modelsignature
        overall_best_accuracy = best_accuracy 
        
    if best_roc_auc > overall_best_roc_auc:
        best_models['auc'] = (best_loss, best_accuracy, best_roc_auc), train_loss, val_loss, modelsignature
        overall_best_roc_auc = best_roc_auc
        


# In[ ]:


print("Results on Test Data Set:")
print(f"Best loss={overall_best_loss}, best accuracy={overall_best_accuracy}, and best AUC={overall_best_roc_auc}")

filename_loss = glob.glob(f"*{best_models['loss'][3]}*loss-{overall_best_loss}*.pth")[0]
filename_acc = glob.glob(f"*{best_models['acc'][3]}*acc-{overall_best_accuracy}*.pth")[0]
filename_auc = glob.glob(f"*{best_models['auc'][3]}*auc-{overall_best_roc_auc}.pth")[0]

print()
print("Files with the best values when evaluated against the test data set:")
print(filename_loss)
print(filename_acc)
print(filename_auc)


# # Boxplot for Loss, Accuracy and AUC

# In[ ]:


results = np.asarray(results)
print(results.shape)
print(results.mean(axis=0), results.std(axis=0),results.max(axis=0))

plt.figure(figsize=(10,10))
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.title(f"Boxplot Results for Model {(hidden_dim, lstm_layers, lr, dropout, dropout_w, dropout_conv)}")
values = ["Loss", "Accuracy", "AUC"]
plt.boxplot(results)
plt.xticks([1, 2, 3], values)
None


# # Evaluate best modell

# In[ ]:


model = LSTM_CNN4(hidden_dim=hidden_dim, lstm_layers=lstm_layers, dropout=0.5, dropout_w=0.5, dropout_conv=0.5)
model.to(device)

calcMetrics(model, dataloader_test, filename_loss, "Test Loss")
roc_auc, targets, outputs = calcMetrics(model, dataloader_test, filename_acc, "Test Accuracy")
calcMetrics(model, dataloader_test, filename_auc, "Test AUC")


# In[ ]:


plotLoss(train_loss, val_loss)


# In[ ]:


plotAUC(targets, outputs)


# In[ ]:




