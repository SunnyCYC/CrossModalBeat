# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 20:12:27 2024
@author: Sunny
"""

#%%
import os
import numpy as np
from pathlib import Path
import tqdm
import glob

import torch
from pm2s.features.crnn_beat_model_v2 import CRNNBeatModel
from pm2s.io.midi_read import read_note_sequence
import pickle

dataset_dirs = {
    ## format:
        ### dname:(main_dir, audio_dir)
    'asap':('./datasets/asap/', 
            './datasets/asap/audio/'),
    }

def getAudioFiles(audio_dir, txt_fname = 'audio_files.txt'):
    audio_paths = []
    audio_list_path = os.path.join(audio_dir, txt_fname)
    with open(audio_list_path, 'r') as file:
        for line in file.readlines():
            audio_paths.append(line.strip('\n'))
    return audio_paths

#%%
def main():
    # cuda_num = int(sys.argv[2])
    # cuda_str = 'cuda:'+str(cuda_num)
    # device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    # Create a beat processor
    model_state_dict_folder = './experiments/pretrained_SBTs/'
    
    for fold_num in np.arange(0, 5):
        model_state_dict_path = os.path.join(model_state_dict_folder, 'fold{}.pt'.format(fold_num))
        model = CRNNBeatModel()
        model.load_state_dict(torch.load(model_state_dict_path, map_location=device))
        model.eval()
        
        
        
        for dname, (dataset_dir, audio_dir) in dataset_dirs.items():
            fail_tracks = []
            print("---"*20)
            print("Processing {} dataset...".format(dname))
            print("---"*20)
            # break
            ### create folder to save symbolic beat estimations
            out_dir = os.path.join(dataset_dir, 'activations', 'SBT_f{}'.format(fold_num))
            if not os.path.exists(out_dir):
                print('Creating folder:{}'.format(out_dir))
                Path(out_dir).mkdir(parents = True, exist_ok = True)
            
            ### folder to load transcribed .mid files
            midi_dir = os.path.join(dataset_dir, 'transcribed-midi')
            midi_files = glob.glob(os.path.join(midi_dir, "*.mid"))
            for midi_file in tqdm.tqdm(midi_files):
                # break
                acti_spath = os.path.join(out_dir, os.path.basename(midi_file).replace('.mid', '.pickle'))
                if not os.path.exists(acti_spath):
                    # print('non exists:{}'.format(acti_spath))
                    try:
                        note_seq = read_note_sequence(midi_file)
                        x = torch.tensor(note_seq).unsqueeze(0).to(device)
                        beat_probs, downbeat_probs = model(x)
                        beat_probs = torch.sigmoid(beat_probs).squeeze(0).detach().numpy()
                        downbeat_probs = torch.sigmoid(downbeat_probs).squeeze(0).detach().numpy()
                        # onsets = note_seq[:, 1]
                        predictions = {'beat-prob': beat_probs, 'downbeat-prob': downbeat_probs, 
                                       'note_seq': note_seq}
                        with open(acti_spath , 'wb') as file:
                            pickle.dump(predictions, file)

                    except:
                        print('fail track:{}'.format(midi_file))
                        fail_tracks.append(midi_file)
            if len(fail_tracks)>0:
                fail_txtpath = os.path.join(out_dir, 'fail_tracks.txt')
                with open(fail_txtpath, 'w') as file:
                    for fail_track in fail_tracks:
                        file.write(fail_track +'\n')
        
        
#%%
if __name__=='__main__':
    main()