# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:49:14 2023

@author: sunnycyc
"""

import librosa
import shutil
import torch
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from .dbeatdataset import AudioBeatTrack, AudioBeatDataset
import tqdm
import math

### Functions for saving best models
def save_checkpoint(
    state, is_best, path, target):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, best_loss = None):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = best_loss
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
#%% 
def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    pbar = tqdm.tqdm(train_loader, disable = False)
    for x, y in pbar:
        # break
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        y_hat = model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
        y_hat = y_hat.reshape((-1, 3))
        y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
        # print('train max:{:.2f}, min: {:.2f}'.format(y_hat.max(), y_hat.min()))
        label_np = y.detach().clone().cpu().numpy()
        class_weights = compute_class_weight(class_weight = "balanced", 
                                             classes= np.unique(label_np), 
                                             y= label_np)

        # weights = [1, 67] # nonbeat, beat, downbeat
        if class_weights.shape[0]<3:
           class_weights = np.array([1, 200, 67])
        class_weights = torch.FloatTensor(class_weights).to(device)
        # class_weights = torch.FloatTensor(weights).to(device)
        CE = nn.CrossEntropyLoss(weight = class_weights)
        loss = CE(y_hat, y)
        loss.backward()
        train_loss += loss
        optimizer.step()

    return train_loss/len(train_loader.dataset)

def valid(model, device, valid_loader ):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            
            y_hat = model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
            y_hat = y_hat.reshape((-1, 3))
            y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
            
            # weights = [1, 67] # nonbeat, beat, downbeat
            label_np = y.detach().clone().cpu().numpy()
            class_weights = compute_class_weight(class_weight = "balanced", 
                                                  classes= np.unique(label_np), 
                                                  y= label_np)
            if class_weights.shape[0]<3:
                class_weights = np.array([1, 200, 67])
            # class_weights = torch.FloatTensor(weights).to(device)
            class_weights = torch.FloatTensor(class_weights).to(device)
            CE = nn.CrossEntropyLoss(weight = class_weights)
            loss = CE(y_hat, y)
            valid_loss += loss
    return valid_loss/len(valid_loader.dataset)


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
#%%
def insertZeros(onsets, probs, FPS =100):
    
    end_frame = math.ceil(onsets[-1]*FPS) ##index of the final frame
    ISz_array = np.zeros((end_frame+1, ))
    frame_ids = []
    for ind, onset in enumerate(list(onsets)):
        # break
        # ISz_array[max(0, round(onset*FPS)-1)] = probs[ind]
        frame_ids.append(round(onset*FPS))
        # ISz_array[round(onset*FPS)] = probs[ind]
    
    
    for f_id in frame_ids:
        # break
        # print('==>　m_id:{}'.format(m_id))
        
        onset_ids = np.where(np.array(frame_ids)==f_id)[0]
        max_prob = probs[onset_ids].max()
        ISz_array[f_id] = max_prob
        # print('onset_ids:{}'.format(onset_ids))
        # if len(onset_ids)>1:
        #     break

    return ISz_array

def insertZeros_onset(onsets, FPS =100):
    
    end_frame = math.ceil(onsets[-1]*FPS) ##index of the final frame
    ISz_array = np.zeros((end_frame+1, ))
    for ind, onset in enumerate(list(onsets)):
        # break
        # ISz_array[max(0, round(onset*FPS)-1)] = 1
        ISz_array[round(onset*FPS)] = 1
    return ISz_array

def resample(xp, fp, step):
    """

    Parameters
    ----------
    xp : np.array
        onsets in second.
    fp : np.array
        symbolic activation values (e.g., beat prob or downbeat prob)
    step : float
        1/fps.

    Returns
    -------
    resampled activation

    """
    
    x = np.arange(0, xp[-1], step = step )
    resampled_x = np.interp(x, xp, fp)
    return resampled_x