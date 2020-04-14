# -*- coding: utf-8 -*-
"""
NISQA Library v0.5

Created on Wed Apr 1 2020

@author: Gabriel Mittag, Quality and Usability Lab, TU Berlin
"""
import os
import datetime
import time

import librosa as lb
import numpy as np
import pandas as pd; pd.options.mode.chained_assignment = None
from tqdm import tqdm

from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#%% Training

def fit_mos_packed(model, model_args, train_opts, train_ds, val_ds, dbs, dev):

    if train_opts['parallel']:
        model = nn.DataParallel(model)
    model.to(dev)

    # Runname and savepath  ---------------------------------------------------
    now_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")
    runname = train_opts['runname'] + '_' + now_str
    model_folder = os.path.join(train_opts['main_folder'], 'models', runname)
    resultspath = os.path.join(train_opts['main_folder'], 'results', runname)
    os.mkdir(model_folder)
    print(runname)

    # Optimizer  -------------------------------------------------------------
    opt = optim.Adam(model.parameters(), lr=train_opts['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            'min',
            verbose=True,
            threshold=0.003,
            patience=train_opts['lr_patience'])
    
    earlyStp = earlyStopper(train_opts['early_stop'])  

    # Dataloader    -----------------------------------------------------------
    train_dl = DataLoader(train_ds,
                          batch_size=train_opts['bs'],
                          shuffle=True,
                          drop_last=False,
                          num_workers=train_opts['num_workers'],
                          pin_memory=False)    
 
    # Start training loop   ---------------------------------------------------
    print('--> start training')
    tic_total = time.time()
    for epoch in range(train_opts['epochs']):

        tic_epoch = time.time()

        # Train model    ------------------------------------------------------
        model.train()
        y_train_hat = np.zeros((len(train_ds), 1))
        batch_cnt = 0
        loss = 0.0

        # Progress bar
        if train_opts['verbose'] == 2:
            pbar = tqdm(iterable=batch_cnt, total=len(train_dl), ascii=">—",
                        bar_format='{bar} {percentage:3.0f}%, {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}')

        for xb_spec_deg, yb_mos, (idx, n_wins) in train_dl:

            # Estimate batch --------------------------------------------------
            xb_spec_deg = xb_spec_deg.to(dev)
            yb_mos = yb_mos.to(dev)
            n_wins = n_wins.to(dev)

            # Forward pass ----------------------------------------------------
            yb_mos_hat = model(xb_spec_deg, n_wins)
            y_train_hat[idx] = yb_mos_hat.detach().cpu().numpy().astype(dtype=float)

            # Loss ------------------------------------------------------------
            lossb = F.mse_loss(yb_mos_hat, yb_mos)
                
            # Backprop
            lossb.backward()
            opt.step()
            opt.zero_grad()

            # Update total loss
            loss += lossb.item()
            batch_cnt += 1

            if train_opts['verbose'] == 2:
                pbar.set_postfix(loss=lossb.item())
                pbar.update()

        if train_opts['verbose'] == 2:
            pbar.close()

        loss = loss/batch_cnt

        # Evaluate  -----------------------------------------------------------            
        train_ds.dfile['y_hat'] = y_train_hat
        db_results_train, r_train = eval_results(
            train_ds.dfile,
            target_mos = train_opts['target_mos'],
            do_print=False,
            do_plot=False)
        
        val_ds.dfile['y_hat'] = predict_mos(model, val_ds, train_opts['eval_bs'], dev, num_workers=train_opts['num_workers'])
        db_results, r = eval_results(
            val_ds.dfile,
            target_mos = train_opts['target_mos'],
            do_print=False,
            do_plot=False)

        # update scheduler ---------------------------------------------------
        scheduler.step(loss)
        early_stp = earlyStp.step(r) 
        
        # Print    ------------------------------------------------------------
        toc_epoch = time.time() - tic_epoch
        if train_opts['verbose'] > 0:
            print('ep {}: sec {:0.0f} es {} lr {:0.0e} loss {:0.4f} r_train {:0.2f} rmse_train {:0.2f} val_r {:0.2f} val_rmse {:0.2f}'
                  .format(epoch+1, toc_epoch, earlyStp.cnt, get_lr(opt), loss, r_train['r_p'], r_train['rmse'], r['r_p'], r['rmse']))

        # Save model    -------------------------------------------------------
        filename = runname + '__' + ('ep_{:03d}'.format(epoch+1)) + '.tar'
        model_path = model_folder + filename
        results = {
            'runname': runname,
            'filename': filename,
            'epoch': epoch+1,
            'loss': loss,
            **r,
            **train_opts,
            'finished': False
            }

        if epoch==0:
            results_hist = pd.DataFrame(results, index=[0])
        else:
            results_hist.loc[epoch] = results

        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
            model_name = model.module.name
        else:
            state_dict = model.state_dict()
            model_name = model.name

        torch_dict = {
            'runname': runname,
            'epoch': epoch+1,
            'train_opts': train_opts,
            'model_args': model_args,
            'model_state_dict': state_dict,
            'optimizer_state_dict': opt.state_dict(),
            'db_results': db_results,
            'results': results,
            'results_hist': results_hist,
            'model_name': model_name,
            'dbs': dbs
            }

        torch.save(torch_dict, model_path)
        results_hist.to_pickle(resultspath+'__results.pd')

        # Early stopping    -----------------------------------------------
        if early_stp:
            print('--> Early stoping best_r_p {:0.2f} best_rmse {:0.2f}'
                  .format(earlyStp.best_r_p, earlyStp.best_rmse))
            results_hist['finished'] = True
            results_hist.to_pickle(resultspath+'__results.pd')
            return results_hist
        
    print('--> training done %0.0f sec' % (time.time() - tic_total))
    results_hist['finished'] = True
    results_hist.to_pickle(resultspath+'__results.pd')
    return results_hist


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class earlyStopper(object):  
    def __init__(self, patience):
        self.best_rmse = 1e10
        self.best_r_p = -1e10
        self.cnt = -1
        self.patience = patience
        
    def step(self, r):
        if r['r_p'] > self.best_r_p:
            self.best_r_p = r['r_p']
            self.cnt = -1   
        if r['rmse'] < self.best_rmse:
            self.best_rmse = r['rmse']
            self.cnt = -1                  
        self.cnt += 1 

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early
        
#%% Models 
class CNN(nn.Module):

    def __init__(self, input_channels, num_k, kernel_size, pool_size, dropout_rate, fc_out_h):
        super().__init__()

        self.name = 'CNN'

        self.input_channels = input_channels
        self.num_k = num_k
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.fc_out_h = fc_out_h

        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.pool_first = nn.MaxPool2d(
                self.pool_size,
                stride = self.pool_size,
                padding = (0,1))

        self.pool = nn.MaxPool2d(
                self.pool_size,
                stride = self.pool_size,
                padding = 0)

        self.conv1 = nn.Conv2d(
                self.input_channels,
                self.num_k,
                self.kernel_size,
                padding = 1)

        self.bn1 = nn.BatchNorm2d( self.conv1.out_channels )

        self.conv2 = nn.Conv2d(
                self.conv1.out_channels,
                2 * self.num_k,
                self.kernel_size,
                padding = 1)

        self.bn2 = nn.BatchNorm2d( self.conv2.out_channels )


        self.conv3 = nn.Conv2d(
                self.conv2.out_channels,
                4 * self.num_k,
                self.kernel_size,
                padding = 1)

        self.bn3 = nn.BatchNorm2d( self.conv3.out_channels )

        self.conv4 = nn.Conv2d(
                self.conv3.out_channels,
                4 * self.num_k,
                self.kernel_size,
                padding = 1)

        self.bn4 = nn.BatchNorm2d( self.conv4.out_channels )

        self.conv5 = nn.Conv2d(
                self.conv4.out_channels,
                4 * self.num_k,
                self.kernel_size,
                padding = 1)

        self.bn5 = nn.BatchNorm2d( self.conv5.out_channels )

        self.conv6 = nn.Conv2d(
                self.conv5.out_channels,
                4 * self.num_k,
                self.kernel_size,
                padding = 1)
        
        self.bn6 = nn.BatchNorm2d( self.conv6.out_channels )
        self.fc = nn.Linear(self.conv6.out_channels *6*2, 768)
        
        if self.fc_out_h:
            self.fc_out = nn.Linear(768, self.fc_out_h)


    def forward(self, x):

        x = F.relu( self.bn1( self.conv1(x) ) )
        x = self.pool_first( x )

        x = F.relu( self.bn2( self.conv2(x) ) )
        x = self.pool( x )

        x = self.dropout(x)
        x = F.relu( self.bn3( self.conv3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.conv4(x) ) )
        x = self.pool( x )

        x = self.dropout(x)
        x = F.relu( self.bn5( self.conv5(x) ) )
        x = self.dropout(x)
        
        x = F.relu( self.bn6( self.conv6(x) ) )
        x = x.view(-1, self.conv6.out_channels *6*2)
            
        x = F.relu(self.fc_out( x ) )

        return x


class LSTM_Attention_SE(nn.Module):
    def __init__(self,
                model_CNN,
                n_cnn_feat=20,
                lstm_h1=20,
                lstm1_num_layers=1,
                lstm1_dropout=0,
                lstm_h2=125,
                lstm2_num_layers=1,
                lstm2_dropout=0,
                bidirectional=True,
                pool_size=1,
                forget_bias=1,
                att_method=None,
                att_h=128,
                post_lstm_dropout=0,
                ):
            
        super().__init__()

        self.name = ['LSTM_Attention_SE', model_CNN.name]

        self.n_cnn_feat = n_cnn_feat
        self.lstm_h1 = lstm_h1
        self.lstm_h2 = lstm_h2
        self.att_h = att_h
       
        self.bidirectional = bidirectional
        self.pool_size = pool_size
        
        self.att_method = att_method
                
        self.max_length = None
        self.forget_bias = forget_bias
        
        self.lstm1_num_layers = lstm1_num_layers
        self.lstm1_dropout = lstm1_dropout

        self.lstm2_num_layers = lstm2_num_layers
        self.lstm2_dropout = lstm2_dropout
        
        self.post_lstm_dropout = post_lstm_dropout
        
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1      
                    
        self.cnn = model_CNN

        self.lstm1 = nn.LSTM(
                input_size = self.n_cnn_feat,
                hidden_size = self.lstm_h1,
                num_layers = self.lstm1_num_layers,
                dropout = self.lstm1_dropout,
                batch_first = True,
                bidirectional = self.bidirectional)
        
        self.lstm2_in = self.num_directions*self.lstm_h1

        self.lstm2 = nn.LSTM(
                input_size = self.lstm2_in,
                hidden_size = self.lstm_h2,
                num_layers = self.lstm2_num_layers,
                dropout = self.lstm2_dropout,
                batch_first = True,
                bidirectional = self.bidirectional)

        if self.att_method=='output_only': 
            self.att = nn.Linear(
                    in_features=self.num_directions*self.lstm_h2,
                    out_features=1)                  
            
            
        elif self.att_method=='luong':
            self.att = Att_Luong(
                    q_dim=self.num_directions*self.lstm_h2, 
                    y_dim=self.num_directions*self.lstm_h2,
                    softmax=False) 
            
        elif self.att_method=='dot':
            self.att = Att_Dot(softmax=False)


        if self.att_method is None:
            self.linear = nn.Linear(
                    in_features=1*self.num_directions*self.lstm2.hidden_size,
                    out_features=1)
        else:
            self.linear = nn.Linear(
                    in_features=2*self.num_directions*self.lstm2.hidden_size,
                    out_features=1)
            
        self.apply_att = Apply_Soft_Attention()

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
          if 'bias' in name:
             if 'lstm' in name:
                 nn.init.constant_(param, 0)
                 nn.init.constant_(param[len(param)//4:len(param)//2], self.forget_bias)
          if 'weight' in name:
              if '_ih_' in name:
                  nn.init.kaiming_normal_(param)
              if '_hh_' in name:
                  nn.init.orthogonal_(param)
                  
                  
    def _mask_attention(self, att, y, n_wins):
        mask = torch.arange(y.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")

    def forward(self, x, n_wins):

        bs = x.shape[0] 
        n_wins = n_wins.view(-1)

        x_packed = pack_padded_sequence(
                x,
                n_wins,
                batch_first=True,
                enforce_sorted=False
                )


        # CNN -----------------------------------------------------------------
        x = self.cnn(x_packed.data)   
        x_packed = PackedSequence(
                x,
                batch_sizes = x_packed.batch_sizes,
                sorted_indices = x_packed.sorted_indices,
                unsorted_indices = x_packed.unsorted_indices)


        # First LSTM ----------------------------------------------------------
        self.lstm1.flatten_parameters()
        x_packed = self.lstm1(x_packed)[0]
        x, _ = pad_packed_sequence(x_packed, 
                                   batch_first=True, 
                                   padding_value=0.0,
                                   total_length=self.max_length)
        
        # Cat and fuse LSTM outputs   -----------------------------------------
        x_packed = pack_padded_sequence(
                x,
                n_wins,
                batch_first=True,
                enforce_sorted=False
                )

        # Second LSTM ---------------------------------------------------------
        self.lstm2.flatten_parameters()
        output, (last_h, _) = self.lstm2(x_packed)
        last_h = last_h.view(self.lstm2_num_layers, self.num_directions, bs, self.lstm_h2)
        last_h = last_h[-1,:].transpose(0,1).contiguous().view(bs, self.num_directions*self.lstm_h2)        
        
        
        # Attention -----------------------------------------------------------
        if self.att_method is None:
            x = last_h     
        
        elif self.att_method=='mean':
            output, _ = pad_packed_sequence(output, 
                                       batch_first=True, 
                                       padding_value=0.0,
                                       total_length=self.max_length)

            output = output.sum(1)
            output = output.div(n_wins.to(torch.float).unsqueeze(1))
            x = torch.cat((output, last_h), 1)   
            
        elif self.att_method=='output_only':
            
            att = self.att(output.data) 
            
            att = PackedSequence(
                    att,
                    batch_sizes = x_packed.batch_sizes,
                    sorted_indices = x_packed.sorted_indices,
                    unsorted_indices = x_packed.unsorted_indices)            
            
            output, _ = pad_packed_sequence(output, 
                                       batch_first=True, 
                                       padding_value=0.0,
                                       total_length=self.max_length)
            
            att, _ = pad_packed_sequence(att, 
                                       batch_first=True, 
                                       padding_value=float("-Inf"),
                                       total_length=self.max_length)  
            
            att = att.transpose(2,1)
            att = F.softmax(att, dim=2)            
            
            output = self.apply_att(output, att).squeeze(1)
            x = torch.cat((output, last_h), 1)            
            
        
        elif self.att_method=='luong':
            
            output, _ = pad_packed_sequence(output, 
                                       batch_first=True, 
                                       padding_value=0.0,
                                       total_length=self.max_length)
            x = last_h.unsqueeze(1)
            att, sim = self.att(x, output)
            self._mask_attention(att, output, n_wins)
            att = F.softmax(att, dim=2)            
            output = self.apply_att(output, att)   
            x = torch.cat((output, x), 2).squeeze(1)    
            
        elif self.att_method=='dot':
            
            output, _ = pad_packed_sequence(output, 
                                       batch_first=True, 
                                       padding_value=0.0,
                                       total_length=self.max_length)
            x = last_h.unsqueeze(1)
            att, sim = self.att(x, output)
            self._mask_attention(att, output, n_wins)
            att = F.softmax(att, dim=2)            
            output = self.apply_att(output, att)   
            x = torch.cat((output, x), 2).squeeze(1)                
                                
        else:
            raise NotImplementedError        
            
        # Predict MOS ---------------------------------------------------------
        x = F.dropout(x, self.post_lstm_dropout)
        y_mos_hat = self.linear(x)
        
        return y_mos_hat
        

#%% Predict
def predict_mos(model, ds, bs, dev, num_workers=0):
    
    dl = DataLoader(ds,
                    batch_size=bs,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False)
    
    model.eval()
    with torch.no_grad():
        y_hat = [[model(xb.to(dev), n_wins).cpu().numpy()] for xb, yb, (idx, n_wins) in dl]

    y_hat = np.concatenate( y_hat, axis=1 ).reshape(-1,1).astype(float)

    return y_hat


#%% Attention
class Att_Dot(torch.nn.Module):
    def __init__(self, softmax=True):
        super().__init__()
        self.softmax = softmax
    def forward(self, query, y):
        att = torch.bmm(query, y.transpose(2,1))
        sim = att.max(2)[0].unsqueeze(1)
        if self.softmax:
            att = F.softmax(att, dim=2)
        return att, sim
    
class Att_Luong(torch.nn.Module):
    def __init__(self, q_dim, y_dim, softmax=True):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.softmax = softmax
        self.W = nn.Linear(self.y_dim, self.q_dim)
    def forward(self, query, y):
        att = torch.bmm(query, self.W(y).transpose(2,1))
        sim = att.max(2)[0].unsqueeze(1)
        if self.softmax:
            att = F.softmax( att, dim=2 )
        return att, sim

class Apply_Soft_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y, att):        
        y = torch.bmm(att, y)       
        return y        

#%% Dataset
class NISQA_Dataset(Dataset):

    def __init__(self, 
                 dfile,
                 dcon=None,
                 mos='MOS',
                 filename='filename',
                 data_dir=None,
                 seg_length=15,
                 max_length=None,
                 to_memory=False,
                 ):

        self.dfile = dfile
        self.dcon = dcon
        self.mos = mos
        self.filename = filename
        self.data_dir = data_dir
        self.seg_length = seg_length
        self.max_length = max_length

        self.to_memory = False
        if to_memory:
            self._to_memory()


    def _to_memory(self):
        self.mem_list = [out for out in tqdm(self, total=len(self)) ]
        self.to_memory = True

    def __getitem__(self, index):
        assert isinstance(index, int), 'index must be integer (no slice)'

        if self.to_memory:
            return self.mem_list[index]
        else:
            # Get MOS
            y_mos = self.dfile[self.mos].iloc[index].reshape(-1).astype('float32')
            
            # Load spec    
            file_path = os.path.join(self.data_dir, self.dfile[self.filename].iloc[index])
            spec = get_librosa_melspec(file_path)  
    
            # Segment specs   
            x_spec_seg, n_wins = segment_specs(spec, self.seg_length, self.max_length)    
        
            return x_spec_seg, y_mos, (index, n_wins)

    def __len__(self):
        return len(self.dfile)


def segment_specs(x, seg_length, max_length=None):
    
    if seg_length % 2 == 0:
        raise ValueError('seg_length must be odd! (seg_lenth={})'.format(seg_length))
    
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    
    n_wins = x.shape[1]-(seg_length-1)

    # broadcast magic to segment melspec
    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(1,0)[idx3,:].unsqueeze(1).transpose(3,2)
    
    # padding
    if max_length:
        xtmp = x
        x = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3])) - 80
        x[:n_wins,:] = xtmp
        
                
    return x, np.array(n_wins)


def get_librosa_melspec(file_path):

    # Calc spec
    y, sr = lb.load(file_path, sr=48000)

    S = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=1024,
        hop_length=480,
        win_length=None,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=1.0,

        n_mels=48,
        fmin=0.0,
        fmax=16e3,
        htk=False,
        norm='slaney',
        )

    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)

    return spec

#%% Evaluate
def eval_results(df, target_mos = 'MOS', pred = 'y_hat',
                 do_print = False,
                 do_plot = False):

    df.db = df.db.astype("category")
    
    # Loop through databases
    results_per_db = pd.DataFrame(columns = ['db', 'r_p', 'r_s', 'rmse'])
    for i in range(len(df.db.cat.categories)):

        # Get dataframe for current database
        db_name = df.db.cat.categories[i]
        df_db = df.loc[df.db==db_name]

        # Get results      
        y = df_db[target_mos].to_numpy()
        y_hat = df_db[pred].to_numpy()
          
        # Calc results
        if np.isnan(y_hat).any() or not np.isfinite(y_hat).any():
            r_p = -1
            r_s = -1
            rmse = -1    
        else: 
            r_p = pearsonr(y, y_hat)[0]
            r_s = spearmanr(y, y_hat)[0]
            rmse = np.sqrt(np.mean((y-y_hat)**2))
            
        # Save results in dataframe
        data = {'db': db_name, 'r_p': r_p, 'r_s': r_s, 'rmse': rmse}
        results_per_db.loc[i] = data

        # Plot
        if do_plot:
            plt.figure(figsize=(4.0, 4.0))
            plt.clf()
            plt.plot(y_hat, y, 'o', label='Original data', markersize=5)
            plt.plot([0, 5], [0, 5], 'k')
            plt.axis([1, 5, 1, 5])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.xticks(np.arange(1, 6))
            plt.yticks(np.arange(1, 6))
            plt.ylabel('Subjective MOS')
            plt.xlabel('Predicted MOS')
            plt.show()
            
        # Print
        if do_print:
            print('%-30s r_p: %0.2f, r_s: %0.2f, rmse: %0.2f'
                  % (db_name+':', r_p, r_s, rmse))

    # Save overall results in dictonary
    r_p_mean = results_per_db.r_p.to_numpy().mean()
    r_s_mean = results_per_db.r_s.to_numpy().mean()
    rmse_mean = results_per_db.rmse.to_numpy().mean()
        
    results_overall = {
        'r_p': r_p_mean,
        'r_s': r_s_mean,
        'rmse': rmse_mean
        }

    return results_per_db, results_overall
