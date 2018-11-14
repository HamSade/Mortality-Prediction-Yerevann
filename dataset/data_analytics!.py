# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:23:28 2018

@author: hamed
"""

import numpy as np
import matplotlib.pyplot as plt

from kk_mimic_dataset import kk_mimic_dataset, loader
from sklearn import datasets

#%% Data loading

log_train_file = "dataset_log/train.log
log_valid_file = "dataset_log/valid.log
log_test_file = "dataset_log/test.log

train_data =   kk_mimic_dataset(phase="train", downsample=False, test=True)
valid_data =   kk_mimic_dataset(phase="valid", downsample=False, test=True)
test_data =   kk_mimic_dataset(phase="test", downsample=False, test=True)

#%% Training data

feature_ind = 0

with open(log_train_file, 'w') as log_tf:
    for sample in train_data:
        x = sample
        log_tf.write('{epoch},{learning_rate},{loss: 8.5f},{AUC:3.3f}\n'.format(
            epoch=epoch_i, learning_rate = lr_, loss=train_loss_, AUC=100*train_auc_))
    
                    
#%% Validation data

with open(log_valid_file, 'w') as log_vf:
    log_vf.write('{epoch},{loss: 8.5f},{AUC:3.3f}\n'.format(
        epoch=epoch_i, loss=valid_loss_, AUC=100*valid_auc_))
    
    
#%% Test data

with open(log_valid_file, 'w') as log_vf:
    log_vf.write('{epoch},{loss: 8.5f},{AUC:3.3f}\n'.format(
        epoch=epoch_i, loss=valid_loss_, AUC=100*valid_auc_))

     
     
#%%  

percent = 20
n_valid = int(percent/100. * 6564)
ind_valid = np.ones(n_valid)
ind_valid = np.concatenate((ind_valid, np.zeros(6564-n_valid)))
ind_valid = np.random.permutation(ind_valid)
ind_test = 1 - ind_valid
ind_valid = np.greater(ind_valid, 0)
ind_test = np.greater(ind_test, 0)
    
def load_dataset(phase): 
    if phase == "train": 
        data_path = "../../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_TRAIN"    
        data = datasets.load_svmlight_file(data_path)           
    else:
        data_path = "../../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_VALID"
        data = np.array(datasets.load_svmlight_file(data_path))
            
        if  phase == "valid":#               
            data = [ data[0][self.ind_valid], data[1][self.ind_valid] ]
        else:            
            data = [ data[0][self.ind_test], data[1][self.ind_test] ]         
    return data
                    
                    
                    
                    
                    
                    