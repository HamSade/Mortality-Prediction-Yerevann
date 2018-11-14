# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:36:39 2018

@author: hamed
"""

from __future__ import absolute_import
from __future__ import print_function

#import numpy as np
import math
import argparse
import os
#import imp
#import re

#from mimic3models import keras_utils
#from mimic3models import common_utils

#from mimic3models import metrics
#from keras.callbacks import ModelCheckpoint, CSVLogger

# Hamed-added :

import torch
import numpy as np
import torch.nn as nn
from time import time
#import pdb
from dataset_loader import dataset_loader

from aae_lstm import AAE, style_disc, cluster_disc
from torch import optim
from torch.autograd import Variable


#%%
parser = argparse.ArgumentParser()

parser.add_argument('--input_size',  type=int, default=256)  #TODO: CHECK
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--output_size', type=int, default=256)
parser.add_argument('--MAX_LENGTH',  type=int, default=100)

#%%
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../data/processed_mimic/in-hospital-mortality'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='/log/')
                    
#%% common_utils.add_common_arguments(parser)
#parser.add_argument('--network', type=str, required=False)
                    
parser.add_argument('--K', type=int, default=5,
                    help='number of clusters')
parser.add_argument('--n_iters', type=int, default=100000,
                    help='number of training iters')
                    
                    
parser.add_argument('--dim', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--depth', type=int, default=1,
                    help='number of bi-LSTMs')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
                    
                    
                    
                    
parser.add_argument('--load_state', type=str, default="",
                    help='state file path')
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--l1', type=float, default=0, help='L1 regularization')
parser.add_argument('--save_every', type=int, default=1,
                    help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="",
                    help='optional prefix of network name')
                    
                    
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--rec_dropout', type=float, default=0.0,
                    help="dropout rate for recurrent connections")


parser.add_argument('--small_part', dest='small_part', action='store_true')
parser.add_argument('--whole_data', dest='small_part', action='store_false')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9,
                    help='beta_1 param for Adam optimizer')

parser.set_defaults(small_part=False)

args = parser.parse_args()
print("args = ", args)

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

#%% testing dataloader

#train_raw, valid_raw = dataset_loader("train", args, target_repl)
#data = train_raw[0]
#labels = train_raw[1]
##print("data = ", data ) 
#print("data.shape = ", data.shape ) 
##print("labels = ", labels )     

#%% Training the Model
# Teacher forcing: http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf
teacher_forcing_ratio = 0.5
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train(x, models, opts, losses):
    
    enc_opt, dec_opt, cluster_disc_opt, style_disc_opt = opts
    ae, c_disc, s_disc = models
    ae_loss, c_loss, s_loss = losses    
    
    target_length = x.size(0)
    
    ## SHOULD BE 3-STEP AAE training    
    enc_opt.zero_grad()
    dec_opt.zero_grad()
    cluster_disc_opt.zero_grad()
    style_disc_opt.zero_grad()
    
    ae_loss = 0
    cluster, style, ae_loss, dec_attn, dec_out = ae(x, ae_loss, ae_loss)
    
    ae_loss.backward()
    enc_opt.step()
    dec_opt.step()
    
    
    # FIRST UPDATE DISCs (AAE PAPER)
    
    ''' In https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae/aae.py
    Authors mistakenly update DFECODER params for generator. Only updating Encoder params make sense here'''
    
    valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)  #TODO: check dimensions
    fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)    
    
    temp = np.random.randint(low=0, high=args.K, size=(1,x.shape[0]) ) #TODO: check the dimension
    categ_batch = np.zeros((temp.size, temp.max()+1))
    categ_batch[np.arange(a.size),temp] = 1
    
    gauss_batch = Variable(Tensor(np.random.rand(args.hidden_size - args.K)), requires_grad=False)  #TODO: check the dimension. STandard Gaussian selected
    
    real_cluster =  c_disc(categ_batch)
    real_style = s_disc(gauss_batch)    
    
    fake_cluster = c_disc(cluster)
    fake_style = s_disc(style)
    
    cluster_real_loss =  c_loss(real_cluster, valid)
    cluster_fake_loss = c_loss(fake_cluster, fake)
    
    real_loss = 
        
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
        
        
    # NEXT UPDATE GENs
    
    
    
    
    
    ae_loss_ = ae_loss.item()/target_length

    return cluster, style, ae_loss_ #, dec_attn, dec_out

#%%
def trainIters(data, models, cluster_disc, style_disc, n_iters, print_every=1000,
               plot_every=100, learning_rate=0.01):
    start = time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
#    optimizer_G = optim.Adam( itertools.chain(encoder.parameters(), decoder.parameters()),
#                                lr=opt.lr, betas=(opt.b1, opt.b2))
#    optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    ae, c_disc, s_disc = models
    
    enc_opt          = optim.SGD(ae.encoder.parameters(), lr=learning_rate)
    dec_opt          = optim.SGD(ae.decoder.parameters(), lr=learning_rate)
    cluster_disc_opt = optim.SGD(cluster_disc.parameters(), lr=learning_rate)
    style_disc_opt   = optim.SGD(style_disc.parameters(), lr=learning_rate)
    
    opts = (enc_opt, dec_opt, cluster_disc_opt, style_disc_opt)    
    
#    ae_criterion = nn.NLLLoss()
    ae_loss = torch.nn.L2Loss()
    c_loss = torch.nn.BCELoss()
    s_loss = torch.nn.BCELoss()
    losses = (ae_loss, c_loss, s_loss)

    for i in range(1, n_iters + 1):
        
        cluster, style, ae_loss, dec_attn, dec_out = train(data, models, opts, losses)
        
        # AE loss plot
        print_loss_total += ae_loss
        plot_loss_total += ae_loss

        if i % print_every == print_every - 1:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if i % plot_every == plot_every - 1:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    
    
#%% Plotting results
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

#%% Training
def main():
    
    # LOADING Data
    train_raw, valid_raw = dataset_loader("train", args, target_repl)
    data = train_raw[0]
    
    #%% Model and device definition

    device = torch.device('cuda' if torch.cuda.device_count() != 0 else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    
    ae = AAE(args.input_size, args.hidden_size, args.output_size, device=device)#.cuda(device=device)
    c_disc = cluster_disc(args.K).cuda(device=device)
    s_disc = style_disc(args.hidden_size - args.K).cuda(device=device)
    
    models = (ae, c_disc, s_disc)
    
    #training
    trainIters(data, models, cluster_disc, style_disc, args.n_iters, print_every=1000,
               plot_every=100, learning_rate=0.01)
    
                            
######################################################################
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
    
def timeSince(since, percent):
    now = time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
                            
#%%
if __name__ == "__main__":
    main()
    