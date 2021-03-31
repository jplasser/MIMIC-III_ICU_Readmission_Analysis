# config file for train.py and test.py
# use MIMIC-III or MIMIC-IV
#
# Set `mimic4` to `True` if you want to evaluate against MIMIC-IV, or to `False` for MIMIC-III.
mimic4 = True

# define the model's hyperparameters here
hidden_dim, lstm_layers, lr, dropout, dropout_w, dropout_conv = (8, 2, 1e-3, 0.3, 0.2, 0.2)

# run for number of epochs...
number_epochs = 100

# path of the preprocessed pickle files
datasetpath = '/system/user/publicwork/student/plasser/MIMIC-III_ICU_Readmission_Analysis/mimic3-readmission/mimic3models/readmission/'
