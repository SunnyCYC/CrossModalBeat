# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:05:52 2024


@author: Sunny
"""

#%%
import os
import numpy as np
from pathlib import Path
import tqdm
import glob
import sys
import torch
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

dataset_dirs = {
    ## format:
        ### dname:(main_dir, audio_dir)
    'asap':('./datasets/asap/', 
            './datasets/asap/audio/'),

    }
    
#%%
def getAudioFiles(audio_dir, txt_fname = 'audio_files.txt'):
    audio_paths = []
    audio_list_path = os.path.join(audio_dir, txt_fname)
    with open(audio_list_path, 'r') as file:
        for line in file.readlines():
            audio_paths.append(line.strip('\n'))
    return audio_paths

def main():
    ### name of the transcription model
    cuda_num = 0
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    
    for dname, (dataset_dir, audio_dir) in dataset_dirs.items():
        print("---"*20)
        print("Processing {} dataset...".format(dname))
        print("---"*20)
        # break
        
        ### create folder to save transcribed .mid files
        out_dir = os.path.join(dataset_dir, 'transcribed-midi')
        if not os.path.exists(out_dir):
            print('Creating folder:{}'.format(out_dir))
            Path(out_dir).mkdir(parents = True, exist_ok = True)
        ### get audiopaths
        audio_paths = getAudioFiles(audio_dir, )
        
        for audio_path in tqdm.tqdm(audio_paths):
            # break
            mid_spath = os.path.join(out_dir, os.path.basename(audio_path).replace('.wav', '.mid'))
            if not os.path.exists(mid_spath):
                (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)
                transcriptor = PianoTranscription(device=device)
                transcribed_dict = transcriptor.transcribe(audio, mid_spath)
#%%
if __name__=='__main__':
    main()
        


