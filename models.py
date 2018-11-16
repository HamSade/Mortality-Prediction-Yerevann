# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:19:12 2018

@author: hamed
"""

# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.SOS_vec = torch.zeros(hidden_size)
        self.device = device
#        self.max_length = MAX_LENGTH
        self.K = K
        
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN( hidden_size, output_size)
    
    ###############################################          
    def forward(self, x, ae_loss, criterion):
        
        in_len = x.shape[0]
        out_len = x.shape[0]
        
        ## Encoder
        enc_outs = torch.zeros(in_len, self.encoder.hidden_size, device=device)        
        enc_hid = self.encoder.initHidden()
        
        for ei in range(in_len):
            enc_out, enc_hid = self.encoder(x[ei], enc_hid)
            enc_outs[ei] = enc_out[0, 0]  #TODO: What?!?
    
        ## AAE trick: Splitting enc output
        cluster = nn.Softmax(enc_hid[:, :self.K], dim=1) #TODO: check dimensions
        style = enc_hid[:, self.K:]
        
        ## Decoder part
        dec_in = torch.tensor([self.SOS_vec], device=device) #[SOS] is a matrix since SOS is a vec. itself
        dec_hid = enc_hid #last state of encoder
        
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
    
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(out_len):
                dec_out, dec_hid, dec_attn = self.decoder(dec_in, dec_hid, enc_outs)
                ae_loss += criterion(dec_out, x[di]) # x as target
                dec_in = x[di]  # x as target =  Teacher forcing    
        else:            
            # Without teacher forcing: use its own predictions as the next input
            for di in range(out_len):
                dec_out, dec_hid, dec_attn = self.decoder(dec_in, dec_hid, enc_outs)                
#                topv, topi = dec_out.topk(1)
#                dec_in = topi.squeeze().detach()  # detach from history as input
                dec_in = dec_out.detach()
                ae_loss += criterion(dec_out, x[di]) #Autoencoding
                
        return cluster, style, ae_loss, dec_attn, dec_out
        
#%% The Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

#        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
#        embedded = self.embedding(input).view(1, 1, -1)
        output = input #embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


#%% The Decoder
######################################################################
### Simple Decoder: use only *context vector*
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)  #TODO:Do we need it in
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

        
#%% Attention Decoder

#MAX_LENGTH = 76 #or 48? Which one is seq len?
#
#class AttnDecoderRNN(nn.Module):
#    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#        super(AttnDecoderRNN, self).__init__()
#        self.hidden_size = hidden_size
#        self.output_size = output_size
#        self.dropout_p = dropout_p
#        self.max_length = max_length
#
##        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#        self.dropout = nn.Dropout(self.dropout_p)
#        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#        self.out = nn.Linear(self.hidden_size, self.output_size)
#
#    def forward(self, input, hidden, encoder_outputs):
##        embedded = self.embedding(input).view(1, 1, -1)
#        embedded = input.view(1, 1, -1)
#        embedded = self.dropout(embedded)
#
#        attn_weights = F.softmax(
#            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                 encoder_outputs.unsqueeze(0))
#
#        output = torch.cat((embedded[0], attn_applied[0]), 1)
#        output = self.attn_combine(output).unsqueeze(0)
#
#        output = F.relu(output)
#        output, hidden = self.gru(output, hidden)
#
#        output = self.out(output[0]) #F.log_softmax(self.out(output[0]), dim=1) #Because we do not need indexes. we need numbers
#
#        return output, hidden, attn_weights
#
#    def initHidden(self):
#        return torch.zeros(1, 1, self.hidden_size, device=device)
