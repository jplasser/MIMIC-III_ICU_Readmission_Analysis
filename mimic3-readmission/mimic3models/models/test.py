#!/usr/bin/env python
# coding: utf-8

import torch

from DataLoader import LoadDataSets
from lstm_cnn import LSTM_CNN4
from lstm_cnn import calcMetrics, plotAUC

# MIMIC value should match the value of the training data
mimic4=True

dataloader_train, dataloader_val, dataloader_test = LoadDataSets(batch_size=64,mimic4=mimic4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Evaluate best model, these values should also match the trained model
hidden_dim, lstm_layers, lr, dropout, dropout_w, dropout_conv = (8, 2, 1e-3, 0.3, 0.2, 0.2)
model = LSTM_CNN4(hidden_dim=hidden_dim, lstm_layers=lstm_layers, dropout=0.5, dropout_w=0.5, dropout_conv=0.5)
model.to(device)

filename_loss = 'model_loss.pth'
filename_acc = 'model_best.pth'
filename_auc = 'model_roc_auc.pth'

calcMetrics(model, dataloader_test, filename_loss, "Test Loss")
calcMetrics(model, dataloader_test, filename_acc, "Test Accuracy")
roc_auc, targets, outputs = calcMetrics(model, dataloader_test, filename_auc, "Test AUC")

plotAUC(targets, outputs)