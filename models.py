# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:19:12 2018

@author: hamed
"""

# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
class cluster_disc(nn.Module):
    def __init__(self, K):
        super(cluster_disc, self).__init__()
    
        self.model = nn.Sequential(
                nn.Linear(K, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
                      )
    
    def forward(self, x):
        out = self.model(x)
        return out

#%%     
class style_disc(nn.Module):
    def __init__(self, d_style):
        super(style_disc, self).__init__()
        
        self.model = nn.Sequential(
                    nn.Linear(d_style, 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                          )
    
    def forward(self, x):
        out = self.model(x)
        return out 
    
#%% AAE_LSTM
class AAE(nn.Module):
    
    def __init__(self, K, input_size, hidden_size, output_size, device):# MAX_LENGTH, 
        
        super(AAE, self).__init__()
        self.teacher_forcing_ratio = 0.5
        self.device = device
        self.K = K
        
        self.encoder = EncoderRNN(input_size, hidden_size).cuda(device=device)
        self.decoder = DecoderRNN( hidden_size, output_size).cuda(device=device)
            
        self.hid_softmax = nn.Softmax(dim=1)
        
    ################################################################################         
    def forward(self, x, ae_loss, criterion):
        
        x = x.float().cuda(device=self.device)
#        in_len = x.shape[1]
        bs = x.shape[0]
        out_len = x.shape[1]
        
        ## Encoder        
#        enc_outs = torch.zeros(in_len, self.encoder.hidden_size, dtype = torch.float, device=self.device)        
        enc_hid = torch.zeros(1, x.shape[0] , self.encoder.hidden_size, device=self.device) #self.encoder.initHidden()
        
        x = x.transpose(0, 1)#.unsqueeze(1) #TODO: Manipulating input to fit GRU (seq_len, batch, input_size)       
#        print("x.shape = ", x.shape) #(48, 1, bs, 76)
        
        enc_outs, enc_hid = self.encoder(x, enc_hid)   
#        print("enc_outs.shape = ", enc_outs.shape) #(48, bs, 76)

        ## AAE trick: Splitting enc output
#        print("enc_hid.shape = ", enc_hid.shape) # (1, bs, 76)
        temp = enc_hid.squeeze() #(bs, 76) # 2D version of last hidden state
        cluster = self.hid_softmax(temp[:, :self.K])
        style = temp[:, self.K:]
        
        ############  Decoder part  ####################################################
        
        SOS = torch.zeros(1, bs, x.shape[-1], device=self.device) #TODO: double check sizes
        dec_in = torch.tensor(SOS, device=device) #Do we need [SOS] (a matrix) as SOS is a vec.?
        dec_hid = enc_hid #last state of encoder (1, bs, 76)
        
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        
        print("use_teacher_forcing = ", use_teacher_forcing)
        
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(out_len):
                dec_out, dec_hid = self.decoder(dec_in, dec_hid)
                print("dec_out.shape", dec_out)
                ae_loss += criterion(dec_out, x[di]) # x as target
                dec_in = x[di]  # x as target =  Teacher forcing    
        else:            
            # Without teacher forcing: use its own predictions as the next input
            for di in range(out_len):
                dec_out, dec_hid = self.decoder(dec_in, dec_hid)                
#                topv, topi = dec_out.topk(1)
#                dec_in = topi.squeeze().detach()  # detach from history as input
                dec_in = dec_out.detach()
                ae_loss += criterion(dec_out, x[di]) #Autoencoding
                
        return cluster, style, ae_loss, dec_out
           
#%% The Decoder

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden):
        output = input
        print("dec_in.shape = ", output.shape)
        output, hidden = self.gru(output, hidden)
#        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
                
#%% The Encoder
                
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

#        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
#        embedded = self.embedding(input).view(1, 1, -1)
        
        output = input#.contiguous().view(1, 1, -1)  #TODO: Manipulating input to fit GRU (seq_len, batch, input_size)
        output, hidden = self.gru(output, hidden)
#        print("GRU output shape = ", output.shape)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
  
  
  
  