# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:36:56 2023

@author: sunnycyc
"""

#%%
import os
from train_modules.dbeatdataset import AudioBeatTrack, AudioBeatDataset

import numpy as np
from pathlib import Path
import tqdm
import sys
import torch
from torch.utils.data import DataLoader
import tqdm
import torch
import torch.nn as nn
from models.BLSTM_BDB_Model import RNNDownBeatProc
import train_modules.utils as utils
import time
import json
from sklearn.utils.class_weight import compute_class_weight

#### set up feature folder based on audio sampling rate, and feature hopsize

MONO = True
DTYPE = np.float32

from train_modules.lookahead_pytorch import Lookahead
#%%
dataset_dirs = {

    'asap':('./datasets/asap/', 
              ('./datasets/asap/audio/', '.wav'), 
              './datasets/asap/annotations/'), 

    }

def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = {'all':0, 'non-beat':0, 'd-beat':0, 'beat': 0, 
                  'len': len(train_loader.dataset)}
    pbar = tqdm.tqdm(train_loader, disable = False)
    for x, ynb, ydb, yb in pbar:
        # break
        pbar.set_description("Training batch")
        x, ynb, ydb, yb = x.to(device), ynb.to(device), ydb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        
        y_hat = model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
        nbeat_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([0.05]).to(device))(y_hat[0, :, 0].reshape((-1,)), ynb.reshape((-1,)))
        dbeat_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([160]).to(device))(y_hat[0, :, 1].reshape((-1,)), ydb.reshape((-1,)))
        beat_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([60]).to(device))(y_hat[0, :, 2].reshape((-1,)), yb.reshape((-1,)))
        
        
        loss = nbeat_loss + dbeat_loss + beat_loss
        loss.backward()
        #### save training loss
        train_loss['all'] += loss.detach()
        train_loss['non-beat'] +=nbeat_loss.detach()
        train_loss['d-beat'] += dbeat_loss.detach()
        train_loss['beat'] += beat_loss.detach()
        optimizer.step()

    return train_loss

def valid(model, device, valid_loader ):
    model.eval()
    valid_loss =  {'all':0, 'non-beat':0, 'd-beat':0, 'beat': 0, 
                  'len': len(valid_loader.dataset)}
    with torch.no_grad():
        for x, ynb, ydb, yb in valid_loader:
            x, ynb, ydb, yb = x.to(device), ynb.to(device), ydb.to(device), yb.to(device)

            
            y_hat = model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
           
            nbeat_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([0.05]).to(device))(y_hat[0, :, 0].reshape((-1,)), ynb.reshape((-1,)))
            dbeat_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([160]).to(device))(y_hat[0, :, 1].reshape((-1,)), ydb.reshape((-1,)))
            beat_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([60]).to(device))(y_hat[0, :, 2].reshape((-1,)), yb.reshape((-1,)))
            loss = nbeat_loss + dbeat_loss + beat_loss
            
            #### save training loss
            valid_loss['all'] += loss.detach()
            valid_loss['non-beat'] +=nbeat_loss.detach()
            valid_loss['d-beat'] += dbeat_loss.detach()
            valid_loss['beat'] += beat_loss.detach()
            
    return valid_loss


def getAudioPaths(audio_txt):
    audiopaths = []
    with open(audio_txt, 'r') as file:
        for line in file.readlines():
            audiopaths.append(line.strip('\n'))
    return audiopaths

def getABTtracks(audiopaths, feature_dir, ann_dir, audio_rate, feature_hopsize):
    #######################################################################
    # for each audiopath, collect the corresponding feature and annotation, 
    # and init an AudioBeatTrack object
    #######################################################################
    audiobeattrack_list= []
    ### collect beat annpaths based on audiopaths and ann_dir
    # annpaths = []
    ### collect featurepaths
    # featurepaths = []
    for audiopath in tqdm.tqdm(audiopaths):
        # break
        annpath = os.path.join(ann_dir, os.path.basename(audiopath).replace('.flac', '.beats').replace('.wav', '.beats'))
        if not os.path.exists(annpath):
            print("Ann does not exists: {}".format(annpath))
            # Ann does not exists: /rock_bring_the_noise.beats
        # else:
        #     annpaths.append(annpath)
        feature_path = os.path.join(feature_dir, os.path.basename(annpath).replace('.beats', '.npy'))
        if not os.path.exists(feature_path):
            print("npy does not exists: {}".format(feature_path))
            # pass
        # else:
        #     featurepaths.append(feature_path)
        #######################################################################
        # init AudioBeatTrack
        #######################################################################
        if os.path.exists(audiopath) and os.path.exists(annpath):
            audiobeattrack = AudioBeatTrack(audiopath, annpath, feature_dir, ann_dir, 
                                            audio_rate = audio_rate, 
                                            feature_hopsize = feature_hopsize )
            audiobeattrack_list.append(audiobeattrack)
    return audiobeattrack_list

def main():
    exp_setting = {
        'sampling-rate': 44100, 
        'feature-hopsize': 441, 
        'feature-type': 'madmomRNNDB', 
        'feature-folder': 'dbeat-features',
        # 'segment-duration': 'full', #seconds
        'train-max-epoch': 200, 
        
        'learning-rate': 1e-3, 
        'patience': 20, 
        'optimizer': 'Lookahead-Adam', 
        
        }
    #### set up feature folder based on audio sampling rate, and feature hopsize
    SAMPLING_RATE = exp_setting['sampling-rate']
    FEATURE_HOPSIZE = exp_setting['feature-hopsize'] ### use madmom setting first
    ### feature type
    Feature_type = exp_setting['feature-type'] ## may also be beatNet, or madmomRNNDB, ....
    ## using 6 seconds to train
    
    patience = exp_setting['patience']
    train_epochs = exp_setting['train-max-epoch']
    
    main_dir = './'
    cuda_num = 0#int(sys.argv[1])
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    

    for fold_num in np.arange(0, 5):
        # must assign
        lr = exp_setting['learning-rate']
        date = '2025-04-30-BLSTM-ABT_f'
       
        exp_name = date +str(fold_num)
        # SEG_DUR = None
        exp_dir = os.path.join(main_dir, 'experiments', exp_name)
        target_jsonpath = exp_dir
        exp_setting['experiment-folder'] = exp_dir
        exp_setting['main-exe-dir'] = main_dir
        
        
        #######################################################################
        ### collect audiobeattrack objects from each dataset
        #######################################################################
    
        train_abt_list = []
        valid_abt_list = []
        for dname, (dataset_dir, (audio_dir, f_end), ann_dir) in dataset_dirs.items():
            # print(dname, '\n', dataset_dir, '\n', audio_dir, f_end, '\n', ann_dir,'\n' )
            # break
            print("==="*20)
            print("Processing {} dataset...".format(dname))
            print("==="*20)
            #######################################################################
            ### collect audiopaths based on audio_files.txt in each audio_dir
            #######################################################################
            # audio_txt = os.path.join(audio_dir, "audio_files.txt")
            
            train_txt = os.path.join(dataset_dir, 'train-info', 'fold-{}-{}.txt'.format(fold_num, 'train'))
            train_audiopaths = getAudioPaths(train_txt)
            valid_txt = os.path.join(dataset_dir, 'train-info', 'fold-{}-{}.txt'.format(fold_num, 'valid'))
            valid_audiopaths = getAudioPaths(valid_txt)
            
            #######################################################################
            ### setup folder directory for beat features, and collect featurepaths
            #######################################################################
            feature_location = os.path.join(dataset_dir,  exp_setting ['feature-folder'])
            feature_folder = Feature_type + '_SR' + str(SAMPLING_RATE)+'_HOP'+str(FEATURE_HOPSIZE)
            feature_dir = os.path.join(feature_location, feature_folder)
            if not os.path.exists(feature_dir):
                print("---> Creating feature folder: {}".format(feature_dir))
                Path(feature_dir).mkdir(parents = True, exist_ok = True)
            
            train_abts = getABTtracks(train_audiopaths, feature_dir, ann_dir, SAMPLING_RATE, FEATURE_HOPSIZE)
            valid_abts = getABTtracks(valid_audiopaths, feature_dir, ann_dir, SAMPLING_RATE, FEATURE_HOPSIZE)
            
            print('---'*20)
            print('train_tracks: {}, valid_tracks: {}'.format(len(train_abts), len(valid_abts)))
            print('---'*20)
            train_abt_list += train_abts
            valid_abt_list += valid_abts
            
            
        #######################################################################
        ### init train/valid dataloaders
        #######################################################################
        trainset = AudioBeatDataset(train_abt_list, 
                                    expand_label = True) 
        train_loader = DataLoader( trainset, batch_size = 1, shuffle = True)
        validset = AudioBeatDataset(valid_abt_list, 
                                    expand_label = True)
        valid_loader = DataLoader( validset, batch_size = 1, shuffle = True)
        
        #######################################################################
        ### information to save
        #######################################################################
        if not os.path.exists(exp_dir):
            print('Creating folder:{}'.format(exp_dir))
            Path(exp_dir).mkdir(parents = True, exist_ok = True)
            
        model_type = 'bdb_blstm'
        model_simpname = 'bdb-blstm_f{}'.format(fold_num)
        model_dir = exp_dir
        model_info = dict(model_type = model_type,
                          model_simpname = model_simpname,
                          model_dir = model_dir, 
                          ### batch size, gpu, date
                          )
        model_params = dict(feature_size = 314, blstm_hidden_size = 25,
                                    nb_layers =3, 
                                    bidirectional = True, 
                                    dropout = 0.1,
                                    out_features = 3,
            )
        model = RNNDownBeatProc(**model_params)
        model.cuda(cuda_num)
        
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay= 0.00001
            )
    
        optimizer = Lookahead(optimizer)
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.3, 
                patience=80,
                cooldown=10
            )
    
        es = utils.EarlyStopping(patience= patience)
        
        t = tqdm.trange(1, train_epochs +1, disable = False)
        ### train losses
        train_losses = []
        train_beat_losses = []
        train_dbeat_losses = []
        train_nbeat_losses = []
        ### valid losses
        valid_losses = []
        valid_beat_losses = []
        valid_dbeat_losses = []
        valid_nbeat_losses = []
        ### time information
        train_times = []
        lr_change_epoch = []
        best_epoch = 0
        stop_t = 0
        for epoch in t:
            # break
            t.set_description("Training Epoch")
            end = time.time()
            train_loss = train(model, device, train_loader, optimizer)
            valid_loss = valid(model, device, valid_loader)
            
            scheduler.step(valid_loss['all'].item()/valid_loss['len'])
            ### train losses
            train_losses.append(train_loss['all'].item()/train_loss['len'])
            train_beat_losses.append(train_loss['beat'].item()/train_loss['len'])
            train_dbeat_losses.append(train_loss['d-beat'].item()/train_loss['len'])
            train_nbeat_losses.append(train_loss['non-beat'].item()/train_loss['len'])
            ### valid losses
            valid_losses.append(valid_loss['all'].item()/valid_loss['len'])
            valid_beat_losses.append(valid_loss['beat'].item()/valid_loss['len'])
            valid_dbeat_losses.append(valid_loss['d-beat'].item()/valid_loss['len'])
            valid_nbeat_losses.append(valid_loss['non-beat'].item()/valid_loss['len'])
    
            t.set_postfix(
            train_loss=train_loss['all'].item()/train_loss['len'], 
            val_loss=valid_loss['all'].item()/valid_loss['len']
            )
    
            stop = es.step(valid_loss['all'].item()/valid_loss['len'])
    
            if valid_loss['all'].item()/valid_loss['len'] == es.best:
                best_epoch = epoch
                
            utils.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_loss': es.best,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    },
                    is_best=valid_loss['all'].item()/valid_loss['len'] == es.best,
                    path=exp_dir,
                    target='RNNDBeatProc'
                )
    
                # save params
            params = {
                    'epochs_trained': epoch,
    
                    'best_loss': es.best,
                    'best_epoch': best_epoch,
                    'train_loss_history': train_losses,
                    'train_beat_loss_history': train_beat_losses, 
                    'train_nbeat_loss_history': train_nbeat_losses, 
                    'train_dbeat_loss_history': train_dbeat_losses,
                    'valid_loss_history': valid_losses,
                    'valid_beat_loss_history': valid_beat_losses, 
                    'valid_nbeat_loss_history': valid_nbeat_losses, 
                    'valid_dbeat_loss_history': valid_dbeat_losses,
                    'train_time_history': train_times,
                    'num_bad_epochs': es.num_bad_epochs,
                    'lr_change_epoch': lr_change_epoch,
                    'stop_t': stop_t,
                    'model_info': model_info,
                    'model_params': model_params, 
                    'exp_setting': exp_setting,
                }
    
            with open(os.path.join(target_jsonpath,  'RNNbeat' + '.json'), 'w') as outfile:
                outfile.write(json.dumps(params, indent=4, sort_keys=True))
    
            train_times.append(time.time() - end)
            
            if stop:
                print("Apply Early Stopping and retrain")
                stop_t +=1
                if stop_t >=5:
                    break
                lr = lr*0.2
                lr_change_epoch.append(epoch)
                optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay= 0.00001
                    )
            
                optimizer = Lookahead(optimizer)
    
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=0.3, 
                        patience=80,
                        cooldown=10
                    )
    
                es = utils.EarlyStopping(patience= patience, best_loss = es.best)
                
#%%
if __name__ == "__main__":
    main()
#%%
