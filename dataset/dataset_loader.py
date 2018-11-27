# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:40:35 2018
@author: hamed 
"""
try:
    import torch
    from torch.utils import data
except:
    pass

#%%
#from Reader import Reader
from Reader import InHospitalMortalityReader as HMR

#%%
class dataset(HMR):
    
    def __init__(self, phase="train"):
         
        self.main_dir = "/home/hamed/Documents/research/data/processed_mimic/in-hospital-mortality"
        self.train_dir = self.main_dir + "/train"
        self.test_dir = self.main_dir + "/test"
        
        self.train_listfile = self.main_dir + "/train_listfile.csv"
        self.valid_listfile = self.main_dir + "/val_listfile.csv"
        self.test_listfile =  self.main_dir + "/test_listfile.csv"
    
#        super(self, dataset).__init__(dataset_dir = self.main_dir)
        
        if phase == "train":
            self.dataset_dir = self.train_dir
            self.listfile = self.train_listfile
        elif phase == "valid":
            self.dataset_dir = self.train_dir
            self.listfile = self.valid_listfile
        else:
            self.dataset_dir = self.test_dir
            self.listfile = self.test_listfile
    
        self._data = HMR(self.dataset_dir, self.listfile, period_length=48.0)

    def __len__(self):        
        return self._data.get_number_of_examples()

    def __getitem__(self, idx):
        return self.read_example(idx)
        
#%%               
#class Reader(object):
#    def __init__(self, dataset_dir, listfile=None):
#        self._dataset_dir = dataset_dir
#        self._current_index = 0
#        if listfile is None:
#            listfile_path = os.path.join(dataset_dir, "listfile.csv")
#        else:
#            listfile_path = listfile
#        with open(listfile_path, "r") as lfile:
#            self._data = lfile.readlines()
#        self._listfile_header = self._data[0]
#        self._data = self._data[1:]
#
#    def get_number_of_examples(self):
#        return len(self._data)
#
#    def random_shuffle(self, seed=None):
#        if seed is not None:
#            random.seed(seed)
#        random.shuffle(self._data)
#    def read_next(self):
#        return self.read_example(to_read_index)  


#%% Testing dataset
#import matplotlib.pyplot as plt
#import numpy as np
#
#dataset_ =  dataset(phase="train")
#
#print('dataset size = ', len(dataset_))
##dataset_.random_shuffle()
#data__ = dataset_._data.read_example(14679) #max test is 3235
#
#X = data__['X']
#t = data__ ['t']
#y = data__ ['y']
#header = data__ ['header']
#name = data__ ['name']
#
#print("X =", X)
#
#print("X shape = ", X.shape)
##print("header = ", header)
#
#X = np.array(X)
#X[X=='None']='nan'
#X[X=='']='nan'
#X[X=='Flex-withdraws']='nan'
#X[X=='No Response-ETT']='nan'
#X[X=='Obeys Commands']='nan'
#X[X=='Oriented']='nan'
#X[X=='Confused']='nan'
##X[X=='No Response-ETT']='0'
#
#
#X = X.astype(np.float)
#
#plt.figure()
#feat_idx = 2
#print(X[:,feat_idx])
#plt.stem(X[:,feat_idx])
#plt.title(header[feat_idx])

#i = 28; j=9
##if X[i,j] == '':
##    X[i,j] = -0.01
#print("X({}, {}) = {}".format(i, j, X[i][j]) )
#print(" y  = ", data__['y'])
#header = data__['header']
#print("header [{}] = {}".format(j, header[j]))
##print("len of header = ", len(header))
#print("name =", data__['name'])

#%% Data loader
def loader(dataset__, batch_size=64, shuffle=True, num_workers=0):
    if shuffle:
        dataset__._data.random_shuffle()
    params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers':num_workers}
    return data.DataLoader(dataset__._data, **params) #, collate_fn=collate_fn_temp)
    
#%% Testing loader

#loader_  = loader(dataset_)
#
#for x in loader_:
#    print(x)

                  
                  
                  
                  
  
 