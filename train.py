# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:36:39 2018

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
parser = argparse.ArgumentParser()

parser.add_argument('--input_size',  type=int, default=76)  #TODO: CHECK
parser.add_argument('--hidden_size', type=int, default=76) #TODO: or 256? Can be changed
parser.add_argument('--output_size', type=int, default=76)

#parser.add_argument('--MAX_LENGTH',  type=int, default=100)

#%%
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../data/processed_mimic/in-hospital-mortality'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='/log/')                                       
                    
#%% common_utils.add_common_arguments(parser)
#parser.add_argument('--network', type=str, required=False)

# Added during running                     
parser.add_argument('--K', type=int, default=5,
                    help='number of clusters')
parser.add_argument('--n_iters', type=int, default=100000,
                    help='number of training iters')
parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")  
parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to '
                             'use one of the provided ones.')  
parser.add_argument('--imputation', type=str, default='previous')

#####################################################################                             
                                        
parser.add_argument('--dim', type=int, default=76, #TODO: was 256. Changed to 76 top match input.shape[-1]
                    help='number of hidden units')
parser.add_argument('--depth', type=int, default=1,
                    help='number of bi-LSTMs')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
                    
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9,
                    help='beta_1 param for Adam optimizer')
parser.add_argument('--beta_2', type=float, default=0.999,
                    help='beta_2 param for Adam optimizer') 
                    
                    
########################                                                        
parser.add_argument('--load_state', type=str, default="",
                    help='state file path')
parser.add_argument('--batch_size', type=int, default=16)

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
                  
                 
#########################
parser.set_defaults(small_part=False)

args = parser.parse_args()
print("args = ", args)

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')   

#%% Training the Model
# Teacher forcing: http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#%%
def train(dl, models, opts, losses):
    
    enc_opt, dec_opt, c_disc_opt, s_disc_opt = opts
    ae, c_disc, s_disc = models
    ae_loss, adv_loss = losses    

    x = next(iter(dl))
    
#    print("train data shape = ", x[0].shape) # bs x 48 x 76
#    x = x[0] #just using the training data  #TODO: when we have validatioon data too
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
    
#%%
def trainIters(dl, models, n_iters, print_every=100,
               plot_every=100):
    start = time()
    plot_losses = []
    print_loss_ae = 0  # Reset every print_every
    plot_loss_ae = 0  # Reset every plot_every
    
#    optimizer_G = optim.Adam( itertools.chain(encoder.parameters(), decoder.parameters()),
#                                lr=args.lr, betas=(opt.b1, opt.b2))
#    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(opt.b1, opt.b2))

    ae, c_disc, s_disc = models
  
    # SGD  
    enc_opt = optim.SGD(ae.encoder.parameters(), lr=args.lr)
    dec_opt = optim.SGD(ae.decoder.parameters(), lr=args.lr)  
    c_disc_opt = optim.SGD(c_disc.parameters(), lr=args.lr)
    s_disc_opt = optim.SGD(s_disc.parameters(), lr=args.lr)
    
    # ADAM
#    b1 = 0.5; b2 = 0.999
#    enc_opt    = optim.Adam( ae.encoder.parameters(), lr=args.lr, betas=(b1, b2))
#    dec_opt    = optim.Adam( ae.decoder.parameters(), lr=args.lr, betas=(b1, b2))
#    c_disc_opt = optim.Adam( c_disc.parameters(), lr=args.lr, betas=(b1, b2))
#    s_disc_opt = optim.Adam( s_disc.parameters(), lr=args.lr, betas=(b1, b2))
        
    opts = (enc_opt, dec_opt, c_disc_opt, s_disc_opt)    
    
#    ae_criterion = nn.NLLLoss()
    ae_loss = torch.nn.MSELoss() #torch.nn.L2Loss()
    adv_loss = torch.nn.BCELoss()
    losses = (ae_loss, adv_loss)
    best_cdl_sofar = np.inf
    print_loss_ae = 0
    plot_loss_ae  = 0
    
    # Just to remove previous logs 
    log_train_file = "log/train.log"
    with open(log_train_file, 'a') as log_tf:
        pass
        
    for i in range(1, n_iters + 1):
        
        if i % 50 == 0:
            print("iteration = ", i)
        cluster, style, ae_loss, adv_losses = train(dl, models, opts, losses)
        c_disc_loss, s_disc_loss, c_gen_loss, s_gen_loss =  adv_losses
        
        ########################
        # AE loss plot
        print_loss_ae += ae_loss
        plot_loss_ae += ae_loss

        if i % print_every == 0:
            print_loss_avg = print_loss_ae / print_every
            print_loss_ae = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, 1. * i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_ae / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_ae = 0
        
        ########################
        # Adv loss plot
        if i % print_every == 0:
            print("c_disc_loss = {}, s_disc_loss = {}, c_gen_loss = {}, s_gen_loss = {}".format(
            c_disc_loss, s_disc_loss, c_gen_loss, s_gen_loss))
            
        ############ Logging events               
        ae_dict = ae.state_dict()
        c_disc_dict = c_disc.state_dict()
        s_disc_dict = s_disc.state_dict()
        checkpoint = {'ae_model': ae_dict,
                      'c_disc_model': c_disc_dict,
                      's_disc_model': s_disc_dict,
                      'args': args,
                      'iteration': i}

        ##### Saving all or best models
        save_path = "saved_models/model"            
        save_mode = "best"

        if save_mode == 'all':
            model_name = save_path + '_ael_{ael:2.4f}_cdl_{cdl:2.6f}_sdl_{sdl:2.6f}_cgn_{cgl:2.4f}_sgl_{sgl:2.4f}.chkpt'.format(
            ael=ae_loss, cdl=c_disc_loss, sdl=s_disc_loss, cgl=c_gen_loss, sgl=s_gen_loss)
            torch.save(checkpoint, model_name)
            
        elif save_mode == 'best':
            model_name = save_path + '.chkpt'
            if c_disc_loss < best_cdl_sofar:
                best_cdl_sofar = c_disc_loss
                torch.save(checkpoint, model_name)
#                print('    - [Info] The checkpoint file has been updated. Best CDL so far is {bcdl:2.6f}'.format(bcdl=best_cdl_sofar))
                    
        # Logging
        if log_train_file:
            with open(log_train_file, 'a') as log_tf:
                
                log_tf.write('{iteration},{ael:2.4f},{cdl:2.6f},{sdl:2.6f},{cgl:2.4f},{sgl:2.4f}\n'.format(
                    iteration=i, ael=ae_loss, cdl=c_disc_loss, sdl=s_disc_loss, cgl=c_gen_loss, sgl=s_gen_loss))
#    showPlot(plot_losses)
    
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
    dataset_ = dataset_class(args, phase="train")
    train_valid_data = data_loader(dataset_, args.batch_size)
    dl = train_valid_data
     
    ### Device
    device = torch.device('cuda' if torch.cuda.device_count() != 0 else 'cpu')    
    ### Model
    ae = AAE(args.K, args.input_size, args.hidden_size, args.output_size, device)#.cuda(device=device)
    c_disc = cluster_disc(args.K).cuda(device=device)
    s_disc = style_disc(args.hidden_size - args.K).cuda(device=device)
    models = (ae, c_disc, s_disc)    
    #training
    trainIters(dl, models, args.n_iters, print_every=1000,
               plot_every=100)
                              
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
    