# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import time
import os
from glob import glob
import datetime
from pathlib import Path

import numpy as np
import pandas as pd; pd.options.mode.chained_assignment=None
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import NISQA_lib as SQ

        
class speechQualityModel(object):
    def __init__(self, args):
        self.args = args
            
        self.runinfos = {}       
        self._getDevice()
        self._loadModel()
        self._loadDatasets()
        self.args['now'] = datetime.datetime.today()
        
        if self.args['mode']=='train':
            print(yaml.dump(self.args, default_flow_style=None, sort_keys=False))
        

    def train(self):
        
        if self.args['dim']==True:
            self._train_dim()
        else:
            self._train_mos()    
            
    def evaluate(self):
        
        if self.args['dim']==True:
            self._evaluate_dim()
        else:
            self._evaluate_mos()      
            
    def predict(self):
        
        if self.args['tr_parallel']:
            self.model = nn.DataParallel(self.model)           
        
        if self.args['dim']==True:
            y_val_hat, y_val = SQ.predict_dim(
                self.model, 
                self.ds_val, 
                self.args['tr_bs_val'],
                self.dev,
                num_workers=self.args['tr_num_workers'])
        else:
            y_val_hat, y_val = SQ.predict_mos(
                self.model, 
                self.ds_val, 
                self.args['tr_bs_val'],
                self.dev,
                num_workers=self.args['tr_num_workers'])                 
            
        self.ds_val.df['model'] = self.args['name']
        
        if self.args['output_dir']:
            self.ds_val.df.to_csv(
                os.path.join(self.args['output_dir'],'NISQA_results.csv'), 
                index=False)
        print(self.ds_val.df.to_string(index=False))
     
    def _train_dim(self):
        '''
        Trains speech quality model.
        '''
        # Initialize  -------------------------------------------------------------
        if self.args['tr_parallel']:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.dev)

        # Runname and savepath  ---------------------------------------------------
        self.runname = self._makeRunname()

        # Optimizer  -------------------------------------------------------------
        opt = optim.Adam(self.model.parameters(), lr=self.args['tr_lr'])        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                'min',
                verbose=True,
                threshold=0.003,
                patience=self.args['tr_lr_patience'])
        earlyStp = SQ.earlyStopper_dim(self.args['tr_early_stop'])      
        
        biasLoss_1 = SQ.biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r']
            )
        
        biasLoss_2 = SQ.biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r_noi']
            )        
        
        biasLoss_3 = SQ.biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r_dis']
            )                
        
        biasLoss_4 = SQ.biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r_col']
            )     

        biasLoss_5 = SQ.biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r_loud']
            )                 

        # Dataloader    -----------------------------------------------------------
        dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args['tr_bs'],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args['tr_num_workers'])
        
        
        # Start training loop   ---------------------------------------------------
        print('--> start training')
        for epoch in range(self.args['tr_epochs']):

            tic_epoch = time.time()
            batch_cnt = 0
            loss = 0.0
            y_mos = self.ds_train.df['mos'].to_numpy().reshape(-1,1)
            y_noi = self.ds_train.df['noi'].to_numpy().reshape(-1,1)
            y_dis = self.ds_train.df['dis'].to_numpy().reshape(-1,1)        
            y_col = self.ds_train.df['col'].to_numpy().reshape(-1,1)    
            y_loud = self.ds_train.df['loud'].to_numpy().reshape(-1,1)          
            y_train = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=1)
            y_train_hat = np.zeros((len(self.ds_train), 5))
                                    
            self.model.train()
            
            
            # Progress bar
            if self.args['tr_verbose'] == 2:
                pbar = tqdm(iterable=batch_cnt, total=len(dl_train), ascii=">—",
                            bar_format='{bar} {percentage:3.0f}%, {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}')
                
            for xb_spec, yb_mos, (idx, n_wins) in dl_train:

                # Estimate batch ---------------------------------------------------
                xb_spec = xb_spec.to(self.dev)
                yb_mos = yb_mos.to(self.dev)
                n_wins = n_wins.to(self.dev)
                
                # Forward pass ----------------------------------------------------
                yb_mos_hat = self.model(xb_spec, n_wins)
                y_train_hat[idx,:] = yb_mos_hat.detach().cpu().numpy()

                # Loss ------------------------------------------------------------                       
                lossb_1 = biasLoss_1.get_loss(yb_mos[:,0].view(-1,1), yb_mos_hat[:,0].view(-1,1), idx)
                lossb_2 = biasLoss_2.get_loss(yb_mos[:,1].view(-1,1), yb_mos_hat[:,1].view(-1,1), idx)
                lossb_3 = biasLoss_3.get_loss(yb_mos[:,2].view(-1,1), yb_mos_hat[:,2].view(-1,1), idx)
                lossb_4 = biasLoss_4.get_loss(yb_mos[:,3].view(-1,1), yb_mos_hat[:,3].view(-1,1), idx)
                lossb_5 = biasLoss_5.get_loss(yb_mos[:,4].view(-1,1), yb_mos_hat[:,4].view(-1,1), idx)
                
                lossb = 1/(4+self.args['dim_mos_w']) * ( 
                    self.args['dim_mos_w'] * lossb_1 +
                    (lossb_2 + lossb_3 + lossb_4 + lossb_5)
                    )
                    
                # Backprop  -------------------------------------------------------
                lossb.backward()
                opt.step()
                opt.zero_grad()

                # Update total loss -----------------------------------------------
                loss += lossb.item()
                batch_cnt += 1

                if self.args['tr_verbose'] == 2:
                    pbar.set_postfix(loss=lossb.item())
                    pbar.update()

            if self.args['tr_verbose'] == 2:
                pbar.close()

            loss = loss/batch_cnt
     
            biasLoss_1.update_bias(y_train[:,0].reshape(-1,1), y_train_hat[:,0].reshape(-1,1))
            biasLoss_2.update_bias(y_train[:,1].reshape(-1,1), y_train_hat[:,1].reshape(-1,1))
            biasLoss_3.update_bias(y_train[:,2].reshape(-1,1), y_train_hat[:,2].reshape(-1,1))
            biasLoss_4.update_bias(y_train[:,3].reshape(-1,1), y_train_hat[:,3].reshape(-1,1))
            biasLoss_5.update_bias(y_train[:,4].reshape(-1,1), y_train_hat[:,4].reshape(-1,1))  
           
                
            # Evaluate   -----------------------------------------------------------
            self.ds_train.df['y_hat_mos'] = y_train_hat[:,0].reshape(-1,1)
            self.ds_train.df['y_hat_noi'] = y_train_hat[:,1].reshape(-1,1)
            self.ds_train.df['y_hat_dis'] = y_train_hat[:,2].reshape(-1,1)
            self.ds_train.df['y_hat_col'] = y_train_hat[:,3].reshape(-1,1)
            self.ds_train.df['y_hat_loud'] = y_train_hat[:,4].reshape(-1,1)
            
            print('--> MOS:')
            db_results_train_mos, r_train_mos = SQ.eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='mos',
                target_ci='mos_ci',
                pred='mos_pred',
                mapping = 'first_order',
                do_print=True
                )
            
            print('--> NOI:')
            db_results_train_noi, r_train_noi = SQ.eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='noi',
                target_ci='noi_ci',
                pred='noi_pred',
                mapping = 'first_order',
                do_print=True
                )            
            
            print('--> DIS:')
            db_results_train_dis, r_train_dis = SQ.eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='dis',
                target_ci='dis_ci',
                pred='dis_pred',
                mapping = 'first_order',
                do_print=True
                )            
            
            print('--> COL:')
            db_results_train_col, r_train_col = SQ.eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='col',
                target_ci='col_ci',
                pred='col_pred',
                mapping = 'first_order',
                do_print=True
                )      
            
            print('--> LOUD:')
            db_results_train_loud, r_train_loud = SQ.eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='loud',
                target_ci='loud_ci',
                pred='loud_pred',
                mapping = 'first_order',
                do_print=True
                )           
            
            SQ.predict_dim(self.model, self.ds_val, self.args['tr_bs_val'], self.dev, num_workers=self.args['tr_num_workers'])
                     
            print('--> MOS:')
            db_results_val_mos, r_val_mos = SQ.eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='mos',
                target_ci='mos_ci',
                pred='mos_pred',
                mapping = 'first_order',
                do_print=True
                )            
            # r_val_mos = {k+'_mos': v for k, v in r_val_mos.items()}
            
            print('--> NOI:')
            db_results_val_noi, r_val_noi = SQ.eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='noi',
                target_ci='noi_ci',
                pred='noi_pred',
                mapping = 'first_order',
                do_print=True
                )           
            r_val_noi = {k+'_noi': v for k, v in r_val_noi.items()}
            
            print('--> DIS:')
            db_results_val_dis, r_val_dis = SQ.eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='dis',
                target_ci='dis_ci',
                pred='dis_pred',
                mapping = 'first_order',
                do_print=True
                )          
            r_val_dis = {k+'_dis': v for k, v in r_val_dis.items()}
            
            print('--> COL:')
            db_results_val_col, r_val_col = SQ.eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='col',
                target_ci='col_ci',
                pred='col_pred',
                mapping = 'first_order',
                do_print=True
                )      
            r_val_col = {k+'_col': v for k, v in r_val_col.items()}
            
            print('--> LOUD:')
            db_results_val_loud, r_val_loud = SQ.eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='loud',
                target_ci='loud_ci',
                pred='loud_pred',
                mapping = 'first_order',
                do_print=True
                )            
            r_val_loud = {k+'_loud': v for k, v in r_val_loud.items()}
            
            r = {
                'train_r_p_mean_con': r_train_mos['r_p_mean_con'],
                 'train_rmse_mean_con': r_train_mos['rmse_mean_con'],
                 'train_rmse_star_map_mean_con': r_train_mos['rmse_star_map_mean_con'],
                 
                'train_r_p_mean_con_noi': r_train_noi['r_p_mean_con'],
                 'train_rmse_mean_con_noi': r_train_noi['rmse_mean_con'],
                 'train_rmse_star_map_mean_con_noi': r_train_noi['rmse_star_map_mean_con'],

                'train_r_p_mean_con_dis': r_train_dis['r_p_mean_con'],
                 'train_rmse_mean_con_dis': r_train_dis['rmse_mean_con'],
                 'train_rmse_star_map_mean_con_dis': r_train_dis['rmse_star_map_mean_con'],

                'train_r_p_mean_con_col': r_train_col['r_p_mean_con'],
                 'train_rmse_mean_con_col': r_train_col['rmse_mean_con'],
                 'train_rmse_star_map_mean_con_col': r_train_col['rmse_star_map_mean_con'],

                'train_r_p_mean_con_loud': r_train_loud['r_p_mean_con'],
                 'train_rmse_mean_con_loud': r_train_loud['rmse_mean_con'],
                 'train_rmse_star_map_mean_con_loud': r_train_loud['rmse_star_map_mean_con'],                 
                 
                 **r_val_mos, **r_val_noi, **r_val_dis, **r_val_col, **r_val_loud, }
            
            
            db_results = {
                'db_results_val_mos': db_results_val_mos,
                'db_results_val_noi': db_results_val_noi,
                'db_results_val_dis': db_results_val_dis,
                'db_results_val_col': db_results_val_col,
                'db_results_val_loud': db_results_val_loud
                          }             
            
            
            # Scheduler update    ---------------------------------------------
            scheduler.step(loss)
            earl_stp = earlyStp.step(r)            

            # Print    --------------------------------------------------------
            ep_runtime = time.time() - tic_epoch
            if self.args['tr_verbose'] > 0:
                print(
                    'ep {} sec {:0.0f} es {} lr {:0.0e} loss {:0.4f} // '
                    'r_p_tr {:0.2f} rmse_tr {:0.2f} rmse3s_tr {:0.2f} // r_p {:0.2f} rmse {:0.2f} rmse3s {:0.2f}  // '
                    'best_r_p {:0.2f} best_rmse {:0.2f},'
                    .format(epoch+1, ep_runtime, earlyStp.cnt, SQ.get_lr(opt), loss, 
                            r['train_r_p_mean_con'], r['train_rmse_mean_con'], r['train_rmse_star_map_mean_con'],
                            r['r_p_mean_con'], r['rmse_mean_con'], r['rmse_star_map_mean_con'],
                            earlyStp.best_r_p, earlyStp.best_rmse))
                
                r_mean = 1/5 * (r['r_p_mean_con'] + 
                          r['r_p_mean_con_noi'] +
                          r['r_p_mean_con_col'] +
                          r['r_p_mean_con_dis'] +
                          r['r_p_mean_con_loud'])
                          
                print('\nAverage r_p {:0.3f}'
                    .format(r_mean)
                    )
                                        
                

            # Save results and model  -----------------------------------------
            self._saveResults(self.model, self.model_args, opt, epoch, loss, ep_runtime, r, db_results)

            # Early stopping    -----------------------------------------------
            if earl_stp:
                print('--> Early stopping. best_r_p {:0.2f} best_rmse {:0.2f}'
                    .format(earlyStp.best_r_p, earlyStp.best_rmse))
                return        

        # Training done --------------------------------------------------------
        print('--> Training done. best_r_p {:0.2f} best_rmse {:0.2f}'
                            .format(earlyStp.best_r_p, earlyStp.best_rmse))        
        return        
        

        
    def _train_mos(self):
        '''
        Trains speech quality model.
        '''
        # Initialize  -------------------------------------------------------------
        if self.args['tr_parallel']:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.dev)

        # Runname and savepath  ---------------------------------------------------
        self.runname = self._makeRunname()

        # Optimizer  -------------------------------------------------------------
        opt = optim.Adam(self.model.parameters(), lr=self.args['tr_lr'])        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                'min',
                verbose=True,
                threshold=0.003,
                patience=self.args['tr_lr_patience'])
        earlyStp = SQ.earlyStopper(self.args['tr_early_stop'])      
        
        biasLoss = SQ.biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r']
            )

        # Dataloader    -----------------------------------------------------------
        dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args['tr_bs'],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args['tr_num_workers'])
        
        # Start training loop   ---------------------------------------------------
        print('--> start training')
        for epoch in range(self.args['tr_epochs']):

            tic_epoch = time.time()
            batch_cnt = 0
            loss = 0.0
            y_train = self.ds_train.df[self.args['csv_mos_train']].to_numpy().reshape(-1)
            y_train_hat = np.zeros((len(self.ds_train), 1))
            self.model.train()
            
            # Progress bar
            if self.args['tr_verbose'] == 2:
                pbar = tqdm(iterable=batch_cnt, total=len(dl_train), ascii=">—",
                            bar_format='{bar} {percentage:3.0f}%, {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}')
                
            for xb_spec, yb_mos, (idx, n_wins) in dl_train:

                # Estimate batch ---------------------------------------------------
                xb_spec = xb_spec.to(self.dev)
                yb_mos = yb_mos.to(self.dev)
                n_wins = n_wins.to(self.dev)

                # Forward pass ----------------------------------------------------
                yb_mos_hat = self.model(xb_spec, n_wins)
                y_train_hat[idx] = yb_mos_hat.detach().cpu().numpy()

                # Loss ------------------------------------------------------------       
                # lossb = F.mse_loss(yb_mos_hat, yb_mos)
                lossb = biasLoss.get_loss(yb_mos, yb_mos_hat, idx)
                    
                # Backprop  -------------------------------------------------------
                lossb.backward()
                opt.step()
                opt.zero_grad()

                # Update total loss -----------------------------------------------
                loss += lossb.item()
                batch_cnt += 1

                if self.args['tr_verbose'] == 2:
                    pbar.set_postfix(loss=lossb.item())
                    pbar.update()

            if self.args['tr_verbose'] == 2:
                pbar.close()

            loss = loss/batch_cnt
            
            biasLoss.update_bias(y_train, y_train_hat)

            # Evaluate   -----------------------------------------------------------
            self.ds_train.df['mos_pred'] = y_train_hat
            db_results_train, r_train = SQ.eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos=self.args['csv_mos_train'],
                target_ci=self.args['csv_mos_train'] + '_ci',
                pred='mos_pred',
                mapping = 'first_order',
                do_print=True
                )
            
            SQ.predict_mos(self.model, self.ds_val, self.args['tr_bs_val'], self.dev, num_workers=self.args['tr_num_workers'])
            db_results, r_val = SQ.eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos=self.args['csv_mos_val'],
                target_ci=self.args['csv_mos_val'] + '_ci',
                pred='mos_pred',
                mapping = 'first_order',
                do_print=True
                )            
            
            r = {'train_r_p_mean_con': r_train['r_p_mean_con'],
                 'train_rmse_mean_con': r_train['rmse_mean_con'],
                 'train_rmse_star_map_mean_con': r_train['rmse_star_map_mean_con'],
                 **r_val}
            
            # Scheduler update    ---------------------------------------------
            scheduler.step(loss)
            earl_stp = earlyStp.step(r)            

            # Print    --------------------------------------------------------
            ep_runtime = time.time() - tic_epoch
            if self.args['tr_verbose'] > 0:
                print(
                    'ep {} sec {:0.0f} es {} lr {:0.0e} loss {:0.4f} // '
                    'r_p_tr {:0.2f} rmse_tr {:0.2f} rmse3s_tr {:0.2f} // r_p {:0.2f} rmse {:0.2f} rmse3s {:0.2f}  // '
                    'best_r_p {:0.2f} best_rmse {:0.2f},'
                    .format(epoch+1, ep_runtime, earlyStp.cnt, SQ.get_lr(opt), loss, 
                            r['train_r_p_mean_con'], r['train_rmse_mean_con'], r['train_rmse_star_map_mean_con'],
                            r['r_p_mean_con'], r['rmse_mean_con'], r['rmse_star_map_mean_con'],
                            earlyStp.best_r_p, earlyStp.best_rmse))

            # Save results and model  -----------------------------------------
            self._saveResults(self.model, self.model_args, opt, epoch, loss, ep_runtime, r, db_results)

            # Early stopping    -----------------------------------------------
            if earl_stp:
                print('--> Early stopping. best_r_p {:0.2f} best_rmse {:0.2f}'
                    .format(earlyStp.best_r_p, earlyStp.best_rmse))
                return        

        # Training done --------------------------------------------------------
        self._logMetricsFinal(self.results_hist)
        print('--> Training done. best_r_p {:0.2f} best_rmse {:0.2f}'
                            .format(earlyStp.best_r_p, earlyStp.best_rmse))        
        return
    
    
    def _evaluate_mos(self):
        print(self.args['csv_mos_val'])
        self.db_results, self.r = SQ.eval_results(
            self.ds_val.df,
            dcon=self.ds_val.df_con,
            target_mos=self.args['csv_mos_val'],
            mapping = 'third_order',
            do_print=True,
            do_plot=True)
    
        print('r_p {:0.2f} rmse {:0.2f} rmse3s {:0.2f}'
            .format(self.r['r_p_mean_con'], self.r['rmse_mean_con'], self.r['rmse_star_map_mean_con'])
            )
    
    
    def _evaluate_dim(self):
        print('--> MOS:')
        self.db_results_val_mos, r_val_mos = SQ.eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='mos',
            target_ci='mos_ci',
            pred='mos_pred',
            mapping = 'first_order',
            do_print=True,
            do_plot=False
            )        
        print('r_p {:0.2f} rmse {:0.2f} rmse3s {:0.2f}'
            .format(r_val_mos['r_p_mean_con'], r_val_mos['rmse_mean_con'], r_val_mos['rmse_star_map_mean_con'])
            )        
        
        # r_val_mos = {k+'_mos': v for k, v in r_val_mos.items()}
        
        print('--> NOI:')
        self.db_results_val_noi, r_val_noi = SQ.eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='noi',
            target_ci='noi_ci',
            pred='noi_pred',
            mapping = 'first_order',
            do_print=True,
            do_plot=False
            )     
        print('r_p {:0.2f} rmse {:0.2f} rmse3s {:0.2f}'
            .format(r_val_noi['r_p_mean_con'], r_val_noi['rmse_mean_con'], r_val_noi['rmse_star_map_mean_con'])
            )             
        r_val_noi = {k+'_noi': v for k, v in r_val_noi.items()}
        
        print('--> DIS:')
        self.db_results_val_dis, r_val_dis = SQ.eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='dis',
            target_ci='dis_ci',
            pred='dis_pred',
            mapping = 'first_order',
            do_print=True,
            do_plot=False
            )   
        print('r_p {:0.2f} rmse {:0.2f} rmse3s {:0.2f}'
            .format(r_val_dis['r_p_mean_con'], r_val_dis['rmse_mean_con'], r_val_dis['rmse_star_map_mean_con'])
            )            
        r_val_dis = {k+'_dis': v for k, v in r_val_dis.items()}
        
        print('--> COL:')
        self.db_results_val_col, r_val_col = SQ.eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='col',
            target_ci='col_ci',
            pred='col_pred',
            mapping = 'first_order',
            do_print=True,
            do_plot=False
            )    
        print('r_p {:0.2f} rmse {:0.2f} rmse3s {:0.2f}'
            .format(r_val_col['r_p_mean_con'], r_val_col['rmse_mean_con'], r_val_col['rmse_star_map_mean_con'])
            )            
        r_val_col = {k+'_col': v for k, v in r_val_col.items()}
        
        print('--> LOUD:')
        self.db_results_val_loud, r_val_loud = SQ.eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='loud',
            target_ci='loud_ci',
            pred='loud_pred',
            mapping = 'first_order',
            do_print=True,
            do_plot=False
            )        
        print('r_p {:0.2f} rmse {:0.2f} rmse3s {:0.2f}'
            .format(r_val_loud['r_p_mean_con'], r_val_loud['rmse_mean_con'], r_val_loud['rmse_star_map_mean_con'])
            )               
        r_val_loud = {k+'_loud': v for k, v in r_val_loud.items()}
        
        self.r = {             
             **r_val_mos, **r_val_noi, **r_val_dis, **r_val_col, **r_val_loud, }            
        
        r_mean = 1/5 * (self.r['r_p_mean_con'] + 
                  self.r['r_p_mean_con_noi'] +
                  self.r['r_p_mean_con_col'] +
                  self.r['r_p_mean_con_dis'] +
                  self.r['r_p_mean_con_loud'])
                  
        print('\nAverage r_p {:0.3f}'
            .format(r_mean)
            )
                

    def _makeRunname(self):
        '''
        Creates individual run name.
        '''        
        runname = self.args['name'] + '_' + self.args['now'].strftime("%y%m%d_%H%M%S%f")
        print('runname: ' + runname)

        return runname
    
 
        
    def _loadDatasets(self):
        if self.args['mode']=='predict_file':
            self._loadDatasetsFile()
        elif self.args['mode']=='predict_dir':
            self._loadDatasetsFolder()  
        elif self.args['mode']=='predict_csv':
            self._loadDatasetsCSVpredict()
        elif self.args['mode']=='train':
            self._loadDatasetsCSV()
        else:
            raise NotImplementedError('Model not available')                        
            
    
    def _loadDatasetsFolder(self):
        data_dir = self.args['deg_dir']
        files = glob( os.path.join(data_dir, '*.wav') )
        files = [os.path.basename(files) for files in files]
        df_val = pd.DataFrame(files, columns=['deg'])
     
        print('# files: {}'.format( len(df_val) ))
        
        # creating Datasets ---------------------------------------------------                        
        self.ds_val = SQ.SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir = data_dir,
            folder_column=None,
            filename_column='deg',
            mos_column=None,              
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = self.args['tr_ds_to_memory'],
            to_memory_workers = self.args['tr_ds_to_memory_workers'],
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = None,                        
            )
        
        
    def _loadDatasetsFile(self):
        data_dir = os.path.dirname(self.args['deg'])
        file_name = os.path.basename(self.args['deg'])        
        df_val = pd.DataFrame([file_name], columns=['deg'])
                
        # creating Datasets ---------------------------------------------------                        
        self.ds_val = SQ.SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir = data_dir,
            folder_column=None,
            filename_column='deg',
            mos_column=None,              
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = self.args['tr_ds_to_memory'],
            to_memory_workers = self.args['tr_ds_to_memory_workers'],
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = None,                        
            )
                
        
    def _loadDatasetsCSVpredict(self):         
        data_dir = self.args['input_dir']
        csv_file_path = os.path.join(self.args['input_dir'], self.args['csv_file'])
        dfile = pd.read_csv(csv_file_path)

        # creating Datasets ---------------------------------------------------                        
        self.ds_val = SQ.SpeechQualityDataset(
            dfile,
            df_con=None,
            data_dir = data_dir,
            folder_column=self.args['csv_deg_dir'],
            filename_column=self.args['csv_deg'],
            mos_column=None,              
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = False,
            to_memory_workers = None,
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = self.args['csv_ref'],                        
            )

        
    def _loadDatasetsCSV(self):        
        data_dir = self.args['input_dir']
        csv_file_path = os.path.join(self.args['input_dir'], self.args['csv_file'])
        dfile = pd.read_csv(csv_file_path)

        if not set(self.args['csv_db_train'] + self.args['csv_db_val']).issubset(dfile.db.unique().tolist()):
            raise ValueError('Not all dbs found in csv')
        
        df_train = dfile[dfile.db.isin(self.args['csv_db_train'])].reset_index()
        df_val = dfile[dfile.db.isin(self.args['csv_db_val'])].reset_index()
        
        if self.args['csv_con'] is not None:
            csv_con_path = os.path.join(self.args['input_dir'], self.args['csv_con'])
            dcon = pd.read_csv(csv_con_path)
            dcon_train = dcon[dcon.db.isin(self.args['csv_db_train'])].reset_index()
            dcon_val = dcon[dcon.db.isin(self.args['csv_db_val'])].reset_index()        
        else:
            dcon = None        
            dcon_train = None        
            dcon_val = None        
        
        print('training size: {}, validation size: {}'.format(len(df_train), len(df_val)))
        
        # creating Datasets ---------------------------------------------------                        
        self.ds_train = SQ.SpeechQualityDataset(
            df_train,
            df_con=dcon_train,
            data_dir = data_dir,
            folder_column=self.args['csv_deg_dir'],
            filename_column=self.args['csv_deg'],
            mos_column=self.args['csv_mos_train'],            
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = self.args['tr_ds_to_memory'],
            to_memory_workers = self.args['tr_ds_to_memory_workers'],
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = self.args['csv_ref'],            
            )

        self.ds_val = SQ.SpeechQualityDataset(
            df_val,
            df_con=dcon_val,
            data_dir = data_dir,
            folder_column=self.args['csv_deg_dir'],
            filename_column=self.args['csv_deg'],
            mos_column=self.args['csv_mos_val'],              
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = self.args['tr_ds_to_memory'],
            to_memory_workers = self.args['tr_ds_to_memory_workers'],
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = self.args['csv_ref'],                        
            )

        self.runinfos['ds_train_len'] = len(self.ds_train)
        self.runinfos['ds_val_len'] = len(self.ds_val)
    
    def _loadModel(self):    
        '''
        Loads the Pytorch models with given input arguments.
        '''   
        
        # if True overwrite input arguments from pretrained model
        if self.args['pretrained_model']:
            if ':' in self.args['pretrained_model']:
                model_path = os.path.join(self.args['pretrained_model'])
            else:
                model_path = os.path.join(os.getcwd(), self.args['pretrained_model'])
            checkpoint = torch.load(model_path, map_location=self.dev)
            
            if self.args['mode']=='train':
                args_new = self.args
                self.args = checkpoint['args']
                self.args['input_dir'] = args_new['input_dir']
                self.args['output_dir'] = args_new['output_dir']
                self.args['csv_file'] = args_new['csv_file']
                self.args['csv_con'] = args_new['csv_con']
                self.args['csv_deg'] = args_new['csv_deg']
                self.args['csv_ref'] = args_new['csv_ref']
                self.args['csv_deg_dir'] = args_new['csv_deg_dir']
                self.args['csv_db_train'] = args_new['csv_db_train']
                self.args['csv_db_val'] = args_new['csv_db_val']            
                self.args['csv_mos_train'] = args_new['csv_mos_train']
                self.args['csv_mos_val'] = args_new['csv_mos_val']                  
                self.args['pretrained_model'] = args_new['pretrained_model']
                
                self.args['tr_epochs'] = args_new['tr_epochs']
                self.args['tr_early_stop'] = args_new['tr_early_stop']
                self.args['tr_bs'] = args_new['tr_bs']
                self.args['tr_bs_val'] = args_new['tr_bs_val']
                self.args['tr_lr'] = args_new['tr_lr']
                self.args['tr_lr_patience'] = args_new['tr_lr_patience']
                self.args['tr_num_workers'] = args_new['tr_num_workers']
                self.args['tr_parallel'] = args_new['tr_parallel']
                self.args['tr_bias_anchor_db'] = args_new['tr_bias_anchor_db']
                self.args['tr_bias_mapping'] = args_new['tr_bias_mapping']
                self.args['tr_bias_min_r'] = args_new['tr_bias_min_r']
                self.args['tr_bias_min_r_noi'] = args_new['tr_bias_min_r_noi']
                self.args['tr_bias_min_r_col'] = args_new['tr_bias_min_r_col']
                self.args['tr_bias_min_r_dis'] = args_new['tr_bias_min_r_dis']
                self.args['tr_bias_min_r_loud'] = args_new['tr_bias_min_r_loud']
                           
                self.args['tr_verbose'] = args_new['tr_verbose']
                
                self.args['tr_ds_to_memory'] = args_new['tr_ds_to_memory']
                self.args['tr_ds_to_memory_workers'] = args_new['tr_ds_to_memory_workers']
                self.args['ms_max_segments'] = args_new['ms_max_segments']
                
            elif self.args['mode']=='predict_file':
                args_new = self.args
                self.args = checkpoint['args']
                self.args['deg'] = args_new['deg']
                self.args['mode'] = args_new['mode']
                self.args['output_dir'] = args_new['output_dir']
                self.args['pretrained_model'] = args_new['pretrained_model']
                
            elif self.args['mode']=='predict_dir':
                args_new = self.args
                self.args = checkpoint['args']
                self.args['deg_dir'] = args_new['deg_dir']
                self.args['mode'] = args_new['mode']
                self.args['output_dir'] = args_new['output_dir']
                self.args['pretrained_model'] = args_new['pretrained_model']   
                if args_new['bs']:
                    self.args['tr_bs_val'] = args_new['bs']
                if args_new['num_workers']:
                    self.args['tr_num_workers'] = args_new['num_workers'] 
                    
            elif self.args['mode']=='predict_csv':
                args_new = self.args
                self.args = checkpoint['args']
                self.args['csv_file'] = args_new['csv_file']
                self.args['mode'] = args_new['mode']
                self.args['output_dir'] = args_new['output_dir']
                self.args['pretrained_model'] = args_new['pretrained_model']   
                self.args['input_dir'] = os.getcwd()
                if args_new['csv_dir'] is None:
                    self.args['csv_deg_dir'] = ''
                else:
                    self.args['csv_deg_dir'] = args_new['csv_dir']
                self.args['csv_deg'] = args_new['csv_deg']
                if args_new['bs']:
                    self.args['tr_bs_val'] = args_new['bs']
                if args_new['num_workers']:
                    self.args['tr_num_workers'] = args_new['num_workers'] 
                    
            else:
                raise NotImplementedError('Mode not available')                        
            
        if self.args['model']=='NISQA_DIM':
            self.args['dim'] = True
        else:
            self.args['dim'] = False
            
        if self.args['model']=='NISQA_DE':
            self.args['double_ended'] = True
        else:
            self.args['double_ended'] = False                        

        # Load Model
        self.model_args = {
            
            'ms_seg_length': self.args['ms_seg_length'],
            'ms_n_mels': self.args['ms_n_mels'],
            
            'cnn_model': self.args['cnn_model'],
            'cnn_c_out_1': self.args['cnn_c_out_1'],
            'cnn_c_out_2': self.args['cnn_c_out_2'],
            'cnn_c_out_3': self.args['cnn_c_out_3'],
            'cnn_kernel_size': self.args['cnn_kernel_size'],
            'cnn_dropout': self.args['cnn_dropout'],
            'cnn_pool_1': self.args['cnn_pool_1'],
            'cnn_pool_2': self.args['cnn_pool_2'],
            'cnn_pool_3': self.args['cnn_pool_3'],
            'cnn_fc_out_h': self.args['cnn_fc_out_h'],
            
            'td': self.args['td'],
            'td_sa_d_model': self.args['td_sa_d_model'],
            'td_sa_nhead': self.args['td_sa_nhead'],
            'td_sa_pool_size': self.args['td_sa_pool_size'],
            'td_sa_pos_enc': self.args['td_sa_pos_enc'],
            'td_sa_num_layers': self.args['td_sa_num_layers'],
            'td_sa_h': self.args['td_sa_h'],
            'td_sa_dropout': self.args['td_sa_dropout'],
            'td_lstm_h': self.args['td_lstm_h'],
            'td_lstm_num_layers': self.args['td_lstm_num_layers'],
            'td_lstm_dropout': self.args['td_lstm_dropout'],
            'td_lstm_bidirectional': self.args['td_lstm_bidirectional'],
            
            'td_2': self.args['td_2'],
            'td_2_sa_d_model': self.args['td_2_sa_d_model'],
            'td_2_sa_nhead': self.args['td_2_sa_nhead'],
            'td_2_sa_pool_size': self.args['td_2_sa_pool_size'],
            'td_2_sa_pos_enc': self.args['td_2_sa_pos_enc'],
            'td_2_sa_num_layers': self.args['td_2_sa_num_layers'],
            'td_2_sa_h': self.args['td_2_sa_h'],
            'td_2_sa_dropout': self.args['td_2_sa_dropout'],
            'td_2_lstm_h': self.args['td_2_lstm_h'],
            'td_2_lstm_num_layers': self.args['td_2_lstm_num_layers'],
            'td_2_lstm_dropout': self.args['td_2_lstm_dropout'],
            'td_2_lstm_bidirectional': self.args['td_2_lstm_bidirectional'],                
            
            'pool': self.args['pool'],
            'pool_output_size': self.args['pool_output_size'],
            'pool_att_h': self.args['pool_att_h'],
            'pool_att_dropout': self.args['pool_att_dropout'],
            }
            
        if self.args['double_ended']:
            self.model_args.update({
                'de_align': self.args['de_align'],
                'de_align_apply': self.args['de_align_apply'],
                'de_align_dim': self.args['de_align_dim'],
                'de_fuse_dim': self.args['de_fuse_dim'],
                'de_fuse': self.args['de_fuse'],        
                })
                        
        if self.args['model']=='NISQA':
            self.model = SQ.NISQA(**self.model_args)     
        elif self.args['model']=='NISQA_DIM':
            self.model = SQ.NISQA_DIM(**self.model_args)     
        elif self.args['model']=='NISQA_DE':
            self.model = SQ.NISQA_DE(**self.model_args)     
        else:
            raise NotImplementedError('Model not available')                        
        
        # Load weights if pretrained model is used ------------------------------------
        if self.args['pretrained_model']:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print('Loaded pretrained model from ' + self.args['pretrained_model'])
            # print('missing_keys:')
            # print(missing_keys)
            # print('unexpected_keys:')
            # print(unexpected_keys)        
            
    def _getDevice(self):
        '''
        Train on GPU if available.
        '''         
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")
    
        if "tr_device" in self.args:
            if self.args['tr_device']=='cpu':
                self.dev = torch.device("cpu")
            elif self.args['tr_device']=='cuda':
                self.dev = torch.device("cuda")            
        print('Device: {}'.format(self.dev))
        

    def _saveResults(self, model, model_args, opt, epoch, loss, ep_runtime, r, db_results):
        '''
        Save model and results in dictionary for every epoch.
        ''' 
        filename = self.runname + '__' + ('ep_{:03d}'.format(epoch+1)) + '.tar'
        run_output_dir = os.path.join(self.args['output_dir'], self.runname)
        model_path = os.path.join(run_output_dir, filename)
        results_path = os.path.join(run_output_dir, self.runname+'__results.csv')
        Path(run_output_dir).mkdir(parents=True, exist_ok=True)

        results = {
            'PartitionKey': self.runname,
            'RowKey': '{:05d}'.format(epoch+1),
            'filename': filename,
            'loss': loss,
            'ep_runtime': '{:0.2f}'.format(ep_runtime),
            **self.runinfos,
            **r,
            **self.args,
            }
        for key in results: 
            results[key] = str(results[key])                        

        if epoch==0:
            self.results_hist = pd.DataFrame(results, index=[0])
        else:
            self.results_hist.loc[epoch] = results

        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
            model_name = model.module.name
        else:
            state_dict = model.state_dict()
            model_name = model.name

        torch_dict = {
            'runname': self.runname,
            'epoch': epoch+1,
            'model_args': model_args,
            'args': self.args,
            'model_state_dict': state_dict,
            'optimizer_state_dict': opt.state_dict(),
            'db_results': db_results,
            'results': results,
            'model_name': model_name,
            }

        torch.save(torch_dict, model_path)
        self.results_hist.to_csv(results_path)
