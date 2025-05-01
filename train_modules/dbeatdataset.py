# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:23:13 2023

@author: Sunny
"""

#%%
import os
from pathlib import Path
from torch.utils.data import Dataset
import tqdm
import numpy as np
import librosa
import glob
from scipy.ndimage import maximum_filter1d

### madmom processors
# https://github.com/CPJKU/madmom/blob/main/madmom/processors.py
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)

from madmom.processors import ( ParallelProcessor, SequentialProcessor)
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor

### Settings
SAMPLING_RATE = 44100
MONO = True
DTYPE = np.float32
RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)

def beat2spec(beats, spec_timesteps = 1000, sr = 44100, hop_length = 441):
    """ purpose:
        1. convert beat information to shape of spectrogram 
        2. convert beat, downbeat, nonbeat to label 2, 1, 0 """
        
    beat_label = np.zeros((spec_timesteps, 1))
    for (beat_time, beat_type) in beats:
#        break
        time_ind = int(beat_time*sr/hop_length)
        if time_ind >= spec_timesteps: ### to exclude case of gtzan with beat ann longer than audio duration
            continue 
        if str(int(beat_type)) == '1':
            beat_label[time_ind, 0] = 1
        else:
            beat_label[time_ind, 0] = 2
    return beat_label

def beat2spec_3D(beats, spec_timesteps = 1000, sr = 44100, hop_length = 441):
    """ purpose:
        1. convert beat information to shape of spectrogram 
        2. save beat, downbeat, nonbeat info to dimension 2, 1, 0 """
        
    beat_label = np.zeros((spec_timesteps, 1))
    dbeat_label = np.zeros((spec_timesteps, 1))
    nonbeat_label = np.ones((spec_timesteps, 1))
    for (beat_time, beat_type) in beats:
        # break
        time_ind = int(beat_time*sr/hop_length)
        if time_ind >= spec_timesteps: ### to exclude case of gtzan with beat ann longer than audio duration
            continue 
        if str(int(beat_type)) == '1':
            dbeat_label[time_ind, 0] = 1
        
        beat_label[time_ind, 0] = 1
        nonbeat_label[time_ind, 0] = 0
    # cat_label = np.hstack([nonbeat_label, beat_label])
    # print(cat_label.shape)
    return nonbeat_label, dbeat_label, beat_label


def widenLabel(beat_frames, size = 3, value = 0.5):
    widen_beat_frame = np.maximum(beat_frames.T, 
                                  maximum_filter1d(beat_frames.T, size=size)*value,).T
    return widen_beat_frame
                                  

class AudioBeatDataset(Dataset):
    
    
    def __init__(self, 
                 audiobeattrack_list,
                 segment_dur_sec = 10, # duration for training
                 # audio_rate = 44100, 
                 # fea_hopsize = 441, 
                 with_downbeat_ann = False,
                 expand_label = False,
                 ):

        self.audiobeattrack_list = audiobeattrack_list
        self.segment_dur_sec = segment_dur_sec
        self.expand_label = expand_label

        
    def __len__(self):
        return len(self.audiobeattrack_list)
    

    def __getitem__(self, index):
        audiobeattrack = self.audiobeattrack_list[index]
        
        return audiobeattrack.get_onehot_data(expand = self.expand_label, )
    
    def __add__(self, other):
        return ConcatAudioBeatDataset([self, other])
    
    def precompute(self, ):
        print("===*20")
        print("Precomputing features...")
        print("===*20")
        for audiobeattrack in tqdm.tqdm(self.audiobeattrack_list):
            audiobeattrack.precompute_feature()

class AudioBeatTrack(object):
    
    def __init__(self, 
                audiopath, 
                annpath,
                feature_folder, 
                ann_folder,
                audio_rate = SAMPLING_RATE, 
                feature_hopsize = 441, 
                # expand_label = False,
                # with_downbeat_ann = False,
                ):
        self.audiopath = audiopath
        self.annpath = annpath
        self.feature_folder = feature_folder
        self.ann_folder = ann_folder
        self.audio_rate = audio_rate
        self.feature_hopsize = feature_hopsize
        self.featurepath = os.path.join(self.feature_folder, 
                                        os.path.basename(self.annpath).replace('.beats', '.npy'))
        self.feature_fps = self.audio_rate/self.feature_hopsize

    
    def get_audio(self):
        audio, rate = librosa.load(self.audiopath, sr = self.audio_rate, mono = MONO, dtype = DTYPE)
        return audio
    
    def get_beatann(self):
        beat_ann_full = np.loadtxt(self.annpath)

        return beat_ann_full
    

        
    
    def get_feature(self):
        if not os.path.exists(self.featurepath):
            self.precompute_feature()
            feature_full = np.load(self.featurepath)
        else:
            feature_full = np.load(self.featurepath)
        
        return feature_full
    
    def precompute_feature(self):
        if os.path.exists(self.featurepath):
            print('exists:{}'.format(os.path.basename(self.featurepath)))
        else:
            if not os.path.exists(self.feature_folder):
                print('Created feature folder: {}'.format(self.feature_folder))
                Path(self.feature_folder).mkdir(parents = True, exist_ok = True)
            ##### calculate madmom feature #####
            
            madmom_preproc = getPreProc(audio_rate = self.audio_rate, 
                                        feature_hopsize = self.feature_hopsize)

            audio = self.get_audio()
            feature_full = madmom_preproc(audio)
            np.save(self.featurepath, feature_full)
    
    def get_data(self, segment_dur_sec = None):
        feature_full = self.get_feature()
        beat_ann_full = self.get_beatann()
      
        
        if segment_dur_sec:
            ### cal feature length (in sec) to decide range to select starting frames
            ####### using Peter's setting
            feature_len = round(segment_dur_sec * (self.audio_rate / self.feature_hopsize )) 
            ###
            feature_dur = feature_full.shape[0]/self.feature_fps 
            
            start_sec = np.random.uniform(0, feature_dur-segment_dur_sec) 

            end_sec = start_sec + (feature_len / self.feature_fps)
            start_frame = int(start_sec*self.feature_fps)

            end_frame = start_frame + feature_len
            ### masking feature
            feature_out = feature_full[start_frame:end_frame, :]
            ### masking beat annotations
            mask = (beat_ann_full[:, 0] >=start_sec) & (beat_ann_full[:, 0] < end_sec)
            beat_masked = beat_ann_full[mask, :]
            beat_masked[:, 0] = beat_masked[:, 0] - start_sec
            # beat_out = beat_masked - start_sec
            return feature_out, beat2spec(beat_masked, spec_timesteps = feature_len, 
                                          sr = self.audio_rate, hop_length = self.feature_hopsize)
        
        else:
            return feature_full, beat2spec(beat_ann_full, spec_timesteps = feature_full.shape[0], 
                                              sr = self.audio_rate, hop_length = self.feature_hopsize)
    def get_onehot_data(self, expand = False ):
        feature_full = self.get_feature()
        beat_ann_full = self.get_beatann()
        # ### only implement full track version
        if not expand:
            nonbeat_label, dbeat_label, beat_label = beat2spec_3D(beat_ann_full, 
                                                  spec_timesteps = feature_full.shape[0], 
                                      sr = self.audio_rate, hop_length = self.feature_hopsize)
            
            return feature_full, nonbeat_label, dbeat_label, beat_label
        else:
            nonbeat_label, dbeat_label, beat_label = beat2spec_3D(beat_ann_full, 
                                                  spec_timesteps = feature_full.shape[0], 
                                      sr = self.audio_rate, hop_length = self.feature_hopsize)
            
        
            return feature_full, nonbeat_label, widenLabel(dbeat_label), widenLabel(beat_label)
                
class ConcatAudioBeatDataset(AudioBeatDataset):
    r"""Concatenate multiple `AudioBeatDataset`s.

    Arguments:
        datasets (list): A list of `AudioBeatsDataset` objects.
    """

    def __init__(self, datasets):
           

        audiobeattrack_list = []
        for dataset in datasets:
            audiobeattrack_list += dataset.audiobeattrack_list

        super().__init__(audiobeattrack_list)
        
################# madmom feature processor


def getPreProc(audio_rate = 44100, feature_hopsize = 441):
    """ returns the madmom features mentioned in the paper"""
    sig = SignalProcessor(num_channels=1, sample_rate=audio_rate)
    multi = ParallelProcessor([])
    frame_sizes = [1024, 2048, 4096]
    num_bands = [3, 6, 12]
    for frame_size, num_bands in zip(frame_sizes, num_bands):
        frames = FramedSignalProcessor(frame_size=frame_size, 
                                       hop_size= feature_hopsize)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
    # stack the features and processes everything sequentially
    pre_processor = SequentialProcessor((sig, multi, np.hstack))
    return pre_processor