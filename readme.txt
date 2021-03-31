## ### ##### ####### ########### ############# ################# ################### #######################
readme.txt, 20210331, v1.0, Jürgen R. Plasser, k08956888
## ### ##### ####### ########### ############# ################# ################### #######################

This is the readme.txt file for the train.py/test.py scripts

Steps to run train.py and test.py:
==================================

0. Preconditions
================

The first prerequisite is the correct Conda environment, so if this environment has been created already, just run this
command:

$ conda activate ICU_Readmission_Analysis_Pytorch_gpu

If you need to create the environment, take environment_pytorch.yml and run:

$ conda env create --file environment_pytorch.yml

There is a second requirement that the pickle files for the preprocessing data have to exist in specific a directory.

For the *.ml.jku.at servers this is the following directory:
/system/user/publicwork/student/plasser/MIMIC-III_ICU_Readmission_Analysis/mimic3-readmission/mimic3models/readmission/

This contains two directories with the three pickle files of the train, validation and test data sets,
respectively for MIMIC-III and MIMIC-IV:

train_data/         - datasets for MIMIC-III
    test_data       - test dataset
    train_data      - train dataset
    val_data        - validation dataset

train_data_mimic4/  - datasets for MIMIC-IV
    test_data       - test dataset
    train_data      - train dataset
    val_data        - validation dataset

Without these files all the preprocessing steps and also the creation of the final datasets in pickle format has to be
computed, which will take some hours of compute.

1. Edit config.py to your requirements, the default values are defined as follows:

# Set `mimic4` to `True` if you want to evaluate against MIMIC-IV, or to `False` for MIMIC-III.
mimic4 = True

# define the model's hyperparameters here
hidden_dim, lstm_layers, lr, dropout, dropout_w, dropout_conv = (8, 2, 1e-3, 0.3, 0.2, 0.2)

# run for number of epochs...
number_epochs = 100

# path of the preprocessed pickle files
datasetpath = '/system/user/publicwork/student/plasser/MIMIC-III_ICU_Readmission_Analysis/mimic3-readmission/mimic3models/readmission/'

2. Run train.py

$ python train.py

(ICU_Readmission_Analysis_Pytorch_gpu) [plasser@raptor MIMIC-III_ICU_Readmission_Analysis]$ python train.py
Training uses MIMIC-IV data.
Loading train, test and validation data... from /system/user/publicwork/student/plasser/MIMIC-III_ICU_Readmission_Analysis/mimic3-readmission/mimic3models/readmission/train_data_mimic4/
Dimensions Train Data:  21578 48 390
Dimensions:  7076 48 390
Dimensions:  7034 48 390
Hyperparameters:
hidden_dim = 8, lstm_layers = 2, lr = 0.001, dropout = 0.3, dropout_w = 0.2, dropout_conv = 0.2

Training is set for maximal number of 5 epochs.
Training Model

Epoch Train: 0, Accuracy Score = 0.6660, Loss = 0.6145
Epoch Val: 0, Accuracy Score = 0.6656 (0.0000), ROCAUC = 0.7535 (0.0000), Loss = 0.6087 (100000.0000)
--------------------
Saving model for best Loss...
Saving model for ROC AUC...
Saving model...
Epoch Train: 1, Accuracy Score = 0.6858, Loss = 0.5942
Epoch Val: 1, Accuracy Score = 0.5893 (0.6656), ROCAUC = 0.7586 (0.7535), Loss = 0.6940 (0.6087)
--------------------
Saving model for ROC AUC...
Epoch Train: 2, Accuracy Score = 0.6890, Loss = 0.5882
Epoch Val: 2, Accuracy Score = 0.6560 (0.6656), ROCAUC = 0.7572 (0.7586), Loss = 0.6336 (0.6087)
--------------------
Epoch Train: 3, Accuracy Score = 0.6950, Loss = 0.5815
Epoch Val: 3, Accuracy Score = 0.7453 (0.6656), ROCAUC = 0.7595 (0.7586), Loss = 0.5371 (0.6087)
--------------------
Saving model for best Loss...
Saving model for ROC AUC...
Saving model...
Epoch Train: 4, Accuracy Score = 0.6981, Loss = 0.5764
Epoch Val: 4, Accuracy Score = 0.6536 (0.7453), ROCAUC = 0.7637 (0.7595), Loss = 0.6492 (0.5371)
--------------------
Saving model for ROC AUC...
Results on validation data set:
...
...
...
Best loss=0.5370655655860901, best accuracy=0.7453363482193329, and best AUC=0.7636773722106676

These are the statedict files with the best values when evaluated against the validation data set:
Filename best loss = model__5_8_2_0.001_0.3-0.2-0.2__epoch-3_loss-0.5370655655860901_acc-0.7453363482193329_auc-0.7595439731271075.pth
Filename best accuracy = model__5_8_2_0.001_0.3-0.2-0.2__epoch-3_loss-0.5370655655860901_acc-0.7453363482193329_auc-0.7595439731271075.pth
Filename best AUC = model__5_8_2_0.001_0.3-0.2-0.2__epoch-4_loss-0.6491642594337463_acc-0.6536178631995477_auc-0.7636773722106676.pth
Training of the model finished.

As results you will find in the current directory the saved state dicts of the training and a plot of the loss curve.
Models are saved when they have best accuracy, best loss or best AUC values for evalutaion against the validation dataset.
Also these these models are saved automatically into files and they will be reused in step 3. with test.py.

The following list of file is an exemplary output:
model__5_8_2_0.001_0.3-0.2-0.2__epoch-0_loss-0.6087297201156616_acc-0.6656302996042962_auc-0.7534701520471845.pth
model__5_8_2_0.001_0.3-0.2-0.2__epoch-1_loss-0.6939568519592285_acc-0.5893159977388355_auc-0.7586488199556662.pth
model__5_8_2_0.001_0.3-0.2-0.2__epoch-3_loss-0.5370655655860901_acc-0.7453363482193329_auc-0.7595439731271075.pth
model__5_8_2_0.001_0.3-0.2-0.2__epoch-4_loss-0.6491642594337463_acc-0.6536178631995477_auc-0.7636773722106676.pth
model_best.pth
model_loss.pth
model_roc_auc.pth
plot-losscurve.png

3. run test.py

$ python test.py

(ICU_Readmission_Analysis_Pytorch_gpu) [plasser@raptor MIMIC-III_ICU_Readmission_Analysis]$ python test.py
Loading train, test and validation data... from /system/user/publicwork/student/plasser/MIMIC-III_ICU_Readmission_Analysis/mimic3-readmission/mimic3models/readmission/train_data_mimic4/
Dimensions Train Data:  21578 48 390
Dimensions:  7076 48 390
Dimensions:  7034 48 390

Test Loss
=========
              precision    recall  f1-score   support

         0.0       0.89      0.77      0.83      5716
         1.0       0.38      0.60      0.46      1318

    accuracy                           0.74      7034
   macro avg       0.63      0.69      0.65      7034
weighted avg       0.80      0.74      0.76      7034

Accuracy Score = 0.738982087005971, Loss = 0.5378969311714172
--------------------
confusion matrix:
[[5221  495]
 [ 844  474]]
accuracy = 0.8096389
precision class 0 = 0.8608409
precision class 1 = 0.48916408
recall class 0 = 0.913401
recall class 1 = 0.3596358
AUC of ROC = 0.7555757551945342
AUC of PRC = 0.43957036776423564
ROC AUC =  0.7555757551945342

Test Accuracy
=============
              precision    recall  f1-score   support

         0.0       0.89      0.77      0.83      5716
         1.0       0.38      0.60      0.46      1318

    accuracy                           0.74      7034
   macro avg       0.63      0.69      0.65      7034
weighted avg       0.80      0.74      0.76      7034

Accuracy Score = 0.738982087005971, Loss = 0.5378969311714172
--------------------
confusion matrix:
[[5221  495]
 [ 844  474]]
accuracy = 0.8096389
precision class 0 = 0.8608409
precision class 1 = 0.48916408
recall class 0 = 0.913401
recall class 1 = 0.3596358
AUC of ROC = 0.7555757551945342
AUC of PRC = 0.43957036776423564
ROC AUC =  0.7555757551945342

Test AUC
========
              precision    recall  f1-score   support

         0.0       0.91      0.63      0.75      5716
         1.0       0.32      0.75      0.45      1318

    accuracy                           0.65      7034
   macro avg       0.62      0.69      0.60      7034
weighted avg       0.80      0.65      0.69      7034

Accuracy Score = 0.6516917827694058, Loss = 0.6536403298377991
--------------------
confusion matrix:
[[4445 1271]
 [ 532  786]]
accuracy = 0.74367356
precision class 0 = 0.8931083
precision class 1 = 0.38210988
recall class 0 = 0.7776417
recall class 1 = 0.5963581
AUC of ROC = 0.7563072694276695
AUC of PRC = 0.43007116982940863
ROC AUC =  0.7563072694276695

The following file is an exemplary output (it plots the last evaluation of the model_auc.pth file):
plot-AUC.png

## ### ##### ####### ########### ############# ################# ################### #######################