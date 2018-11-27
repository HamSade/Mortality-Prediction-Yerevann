# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:57:42 2018

@author: hamed
"""

from __future__ import absolute_import
from __future__ import print_function

import math
import argparse
import os
# Hamed-added :
###########################################
import torch
from torch import optim
from torch.autograd import Variable

import numpy as np
from time import time
from dataset_loader import dataset_class, data_loader

from models import AAE, style_disc, cluster_disc


import pdb
import colored_traceback; colored_traceback.add_hook()

#%%
def test(dl, models, losses):
    
    
    checkpoint = {'ae_model': ae_dict,
                      'c_disc_model': c_disc_dict,
                      's_disc_model': s_disc_dict,
                      'args': args,
                      'iteration': i}
    #######################################
    model_.eval()

    total_loss = 0
    pred = []
    gold = []
    n_seq_total = 0
    
    with torch.no_grad():
        for batch in validation_data:
#                tqdm(validation_data, mininterval=2,
#                desc='  - (Validation) ', leave=False):
            # prepare data
            src_seq, src_pos, gold_, src_fixed_feats = map(lambda x: x.to(device), batch)
            gold_ = gold_.view(-1)
            
            # forward
            pred_ = model_(src_seq, src_pos)
            loss = cal_loss(pred_, gold_, smoothing=False)  #no smoothing in evaluation

            # note keeping
            total_loss += loss.item()
                 
            pred_ = pred_.max(1)[1]
            pred.append(pred_.cpu().numpy())
            gold.append(gold_.cpu().numpy())
            n_seq_total += 1

            # Printing loss
            if n_seq_total%print_chunk_size == print_chunk_size-1:
                print("validation loss = ", loss.item())

    total_loss = total_loss/n_seq_total

    auc_valid = AUCMeter()
    for i in range(len(pred)):
        auc_valid.add(pred[i], gold[i])
    
    auc_ = auc_valid.value()[0] 
    
    return total_loss, auc_
    
    ###########################
    enc_opt, dec_opt, c_disc_opt, s_disc_opt = opts
    ae, c_disc, s_disc = models
    ae_loss, adv_loss = losses    

    x = next(iter(dl))
#        print("train data shape = ", x[0].shape) # bs x 48 x 76
    x = x[0] #just using the training data
    target_length = x.shape[1] #seq length
    
    ## SHOULD BE 3-STEP AAE training (Reconst + Discs + Enc (as Gens))
    enc_opt.zero_grad()
    dec_opt.zero_grad()
    c_disc_opt.zero_grad()
    s_disc_opt.zero_grad()
    
#        ae_loss = 0
    cluster, style, ae_loss_ = ae(x, ae_loss)
    
#        pdb.set_trace()
    
    ae_loss_.backward(retain_graph=True)
    dec_opt.step() #TODO: the order matters
    enc_opt.step()
       
    # FIRST UPDATE DISCs and then GENs (AAE PAPER)    
    ''' In https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae/aae.py
    Authors mistakenly update DECODER params for generator. Only updating Encoder params make more sense here'''
    
    real = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)  #TODO: check dimensions
    fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)    
    
    temp = np.random.randint(low=0, high=args.K, size=(1,x.shape[0]) ) #TODO: check the dimension
    categ_batch = np.zeros((temp.size, args.K))#temp.max()+1))
    categ_batch[np.arange(temp.size),temp] = 1
    categ_batch = Variable(Tensor(categ_batch), requires_grad=False)
    
    gauss_batch = Variable(Tensor(np.random.rand( x.shape[0], int(args.hidden_size - args.K)  ) ), requires_grad=False)  #TODO: STandard Gaussian selected
    
    real_cluster = c_disc(categ_batch)
    real_style   = s_disc(gauss_batch)    
    
    fake_cluster = c_disc(cluster.detach())
    fake_style = s_disc(style.detach())
    
    # Adv losses        
    c_disc_loss = 0.5 * (adv_loss(real_cluster, real) + adv_loss(fake_cluster, fake))
    s_disc_loss = 0.5 * (adv_loss(real_style, real) + adv_loss(fake_style, fake))
    c_gen_loss = adv_loss(fake_cluster, real)      
    s_gen_loss = adv_loss(fake_style, real) 
    
    # Cluster disc
    c_disc_loss.backward(retain_graph=True)
    c_disc_opt.step()
    
    # Style disc
    s_disc_loss.backward(retain_graph=True)
    s_disc_opt.step()
    
    # Cluster gen(encoder)
    c_gen_loss.backward()
    
    # Style gen(encoder)
    s_gen_loss.backward()
    
    enc_opt.step()  #TODO: only one encoder update as the Gen for both cluster and style parts

    ae_loss_final = ae_loss_.item()/target_length
    c_disc_loss_ = c_disc_loss.item()/target_length
    s_disc_loss_ = s_disc_loss.item()/target_length
    c_gen_loss_ = c_gen_loss.item()/target_length
    s_gen_loss_ = s_gen_loss.item()/target_length
    
    adv_losses_ = (c_disc_loss_, s_disc_loss_, c_gen_loss_, s_gen_loss_)

    return cluster, style, ae_loss_final, adv_losses_#, dec_attn, dec_out