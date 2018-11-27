# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:29:43 2018

@author: hamed
"""

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except:
    pass

import numpy as np
import os

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models.in_hospital_mortality import utils

# Hamed-added :
from time import time

#%%
def dataset_reader(phase, args, target_repl=False):
    
    if phase == "train":
        #% Build readers & discretizers
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                                 listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                 period_length=48.0)
        
        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                               listfile=os.path.join(args.data, 'val_listfile.csv'),
                                               period_length=48.0)
        
        discretizer = Discretizer(timestep=float(args.timestep),
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')
        
        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
         
        #%% Data normalization (by mean and variance)        
        normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)        
#        args_dict = dict(args._get_kwargs()) #TODO: reverse
        args_dict = {}
        args_dict['header'] = discretizer_header
        args_dict['task'] = 'ihm'
        args_dict['target_repl'] = target_repl
        
        #%% Read data        
        start = time()
        print("Reading started")
        train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part, return_names=False)
        val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part, return_names=False)
        
        if target_repl:
            T = train_raw[0][0].shape[0]        
            def extend_labels(data):
                data = list(data)
                labels = np.array(data[1])  # (B,)
                data[1] = [labels, None]
                data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
                data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
                return data
        
            train_raw = extend_labels(train_raw)
            val_raw = extend_labels(val_raw)   
        
        print("Reading finished after {} seconds".format(time() - start))        
        return (train_raw, val_raw)
     
    else: ################################### TEST phase
        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
        test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)
        return test_raw

#%%
class dataset_class(Dataset):
    def __init__(self, args, phase="train"):
        self.dataset = dataset_reader(args=args, phase=phase)
        self.phase = phase
        
        if phase == "train":
            # train dataset = (train [data, lables], valid[data, labels])
            self.train_set = self.dataset[0][0] # Labels are discarded
            self.valid_set = self.dataset[1][0]          
#            print("self.train_set.shape, self.valid_set.shape = ", self.train_set.shape, self.valid_set.shape)
        else:
            self.test_set = self.dataset[0]
            
    def __len__(self):
        if self.phase == "train":
            return self.train_set.shape[0]
        else:
            return self.test_set.shape[0]

    def __getitem__(self, idx):
        if self.phase == "train":
            return torch.from_numpy(self.train_set[idx])    #(torch.from_numpy(self.train_set[idx]),torch.from_numpy(self.valid_set[idx]))  #TODO: when we have validatioon data too 
        else:
            return self.test_set[idx]
    
#%% 
def data_loader(dataset__, batch_size = 16 , shuffle=True, num_workers=0):
#    if shuffle:
#        dataset__.data.random_shuffle()
    return DataLoader(dataset__, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers)

#%% New test
#import matplotlib.pyplot as plt
#
#class ARGS():
#    def __init__(self):
#        self.data = os.path.join(os.path.dirname(__file__), '../data/processed_mimic/in-hospital-mortality')
#        self.timestep =  1.0
#        self.normalizer_state = None
#        self.small_part = True
#        self.imputation = "previous"
#        self.batch_size = 16
#        
#args = ARGS()
#
#dataset_ = dataset_class(args, phase="train")
#train_valid_data = data_loader(dataset_, args.batch_size)
#dl = train_valid_data
#
##data__ = dataset_[40]
##train_data_ =  data__[0]
#
#data__ = next(iter(dl))
#print("data__[0].shape = ", data__[0].shape)
#train_data_ =  data__[0]
#valid_data_ = data__[1]
#
#print("len(dataset_) = ", len(dataset_))
#print("len(dl) = ", len(dl))
#print('train_data_.shape = ', train_data_.shape)
#
#print('valid_data_.shape = ', valid_data_.shape)
#
#
#
##%%
#patient_ind = 15
#for fi in range(76):
#    print("fi = ", fi)
#    plt.figure(fi)
#    plt.stem(train_data_[patient_ind, :,fi])
#    plt.show()

  
#%% OLD testing dataloader
#train_raw, valid_raw = data_loader("train", args, target_repl)
#data = train_raw[0]
#labels = train_raw[1]
##print("data = ", data ) 
#print("data.shape = ", data.shape ) 
##print("labels = ", labels )  
     