# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 20:12:27 2024

@author: Sunny
"""

#%%
import os
import train_modules.utils as utils
import numpy as np
from pathlib import Path
import tqdm
import glob

import pickle


dataset_dirs = {
    ## format:
        ### dname:(main_dir, audio_dir)
    'asap':('./datasets/asap/', 
            './datasets/asap/audio/'),
       
    }




#%%
def main():

    for fold_num in np.arange(0, 5):
        interpolation_fps = 100
        # break
        for dname, (dataset_dir, audio_dir) in dataset_dirs.items():
            # fail_tracks = []
            print("---"*20)
            print("Processing {} dataset...".format(dname))
            print("---"*20)
            # break
            raw_symacti_srcs = [os.path.join(dataset_dir, 'activations', 'SBT_f{}'.format(fold_num))]
            
            for raw_symacti_src in tqdm.tqdm(raw_symacti_srcs):
                # break
                print("==="*20)
                print("Processing Symbolic Acti: {}".format(raw_symacti_src))
                print("==="*20)
                new_symacti_type = 'E2F' ## linear interpolation
                dst_symacti_name = '-'.join(os.path.basename(raw_symacti_src).split('-')[:2]+[new_symacti_type])
                out_dir = os.path.join(dataset_dir, 'activations', dst_symacti_name) 
            
                if not os.path.exists(out_dir):
                    print('Creating folder:{}'.format(out_dir))
                    Path(out_dir).mkdir(parents = True, exist_ok = True)

                src_actipaths = glob.glob(os.path.join(raw_symacti_src, "*.pickle"))
                
                for src_actipath in tqdm.tqdm(src_actipaths):
                    # break
                    dst_actipath = os.path.join(out_dir, os.path.basename(src_actipath))
                    if not os.path.exists(dst_actipath):
                        with open(src_actipath, 'rb') as file:
                            src_dict = pickle.load(file)
                        beat_probs = src_dict['beat-prob']
                        dbeat_probs = src_dict['downbeat-prob']
                        onsets = src_dict['note_seq'][:, 1]
                        beat_rs = utils.insertZeros(onsets, beat_probs, interpolation_fps)
                        dbeat_rs = utils.insertZeros(onsets, dbeat_probs, interpolation_fps)
                        save_dict = {'beat-acti': beat_rs, 'downbeat-acti': dbeat_rs, 'interpolation-fps': interpolation_fps}
                        
                        with open(dst_actipath, 'wb') as file:
                            pickle.dump(save_dict, file)

                

        
#%%
if __name__=='__main__':
    main()