# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 08:24:52 2024
   
@author: sunnycyc
"""

#%%
import os
import numpy as np
from pathlib import Path
import tqdm
import glob
import pickle
from scipy.ndimage import gaussian_filter1d

def main():

    fusion_types= ['ADD', 'MTP',] 
    for fusion_type in fusion_types:
        # break
        for fold_num in np.arange(0, 5):
            print(fold_num)
            abt_name = 'ABT_f{}'.format(fold_num)
            sbt_name = 'SBT_f{}-E2F'.format(fold_num)
            print('---'*20)
            print('ABT:{}'.format(abt_name))
            print('SBT:{}'.format(sbt_name))
            print('---'*20)
            # break
            ### set target dataset dirs : name:(main_dir, beat_ann_dir, dbeat_ann_dir,  acti_dir)
            testset_dirs = {
                'asap':('./datasets/asap/', 
                        './datasets/asap/annotations', 
                        './datasets/asap/activations', 
                        ),
                }

            for dataset_name, (main_data_dir, beat_ann_dir, acti_dir) in testset_dirs.items():
                # break
                print("======"*10)
                print("Processing {}".format(dataset_name))
                print("======"*10)
                
                #### create folder for fusion acti
                fusion_acti_dir = os.path.join(main_data_dir, 'activations', fusion_type + '_f{}'.format(fold_num))
                if not os.path.exists(fusion_acti_dir):
                    print('creating folder for fusion:', fusion_acti_dir)
                    Path(fusion_acti_dir).mkdir(parents = True, exist_ok = True)
                
                ### for each sym acti path, check if fusion acti exists, otherwise generate it and save
                sym_acti_paths =  glob.glob(os.path.join(acti_dir, sbt_name, "*.pickle"))
                for sym_acti_path in tqdm.tqdm(sym_acti_paths):
                    # break
                    fus_acti_path = os.path.join(fusion_acti_dir, os.path.basename(sym_acti_path).replace('.pickle', '.npy'))
                    if not os.path.exists(fus_acti_path):
                        ### Load symbolic acti
                        with open(sym_acti_path, 'rb') as file:
                            load_dict = pickle.load(file)
                        sym_beat_activation = load_dict['beat-acti']
                        sym_beat_acti = gaussian_filter1d(sym_beat_activation, sigma=3)
                        sym_beat_acti = sym_beat_acti/sym_beat_acti.max()
                        
                        ### Load audio acti
                        aud_acti_path = os.path.join(acti_dir, abt_name, os.path.basename(sym_acti_path).replace('.pickle', '.npy'))
                        aud_beat_activation = np.load(aud_acti_path)[:, 2]
                        aud_beat_acti = gaussian_filter1d(aud_beat_activation, sigma=3)
                        aud_beat_acti = aud_beat_acti/aud_beat_acti.max()
                        ### Zero-padding 
                        sym_beat_acti_zpad = np.zeros(aud_beat_acti.shape)
                        sym_beat_acti_zpad[:len(sym_beat_acti)] = sym_beat_acti
                        
                        if fusion_type == 'ADD':
                            fusion_acti_sum = sym_beat_acti_zpad + aud_beat_acti
                            fusion_acti_gs = gaussian_filter1d(fusion_acti_sum, sigma=3)
                            fusion_acti = fusion_acti_gs/fusion_acti_gs.max()
                        elif fusion_type =='MTP':
                            fusion_acti_mtp = sym_beat_acti_zpad * aud_beat_acti
                            fusion_acti_mtp_gs = gaussian_filter1d(fusion_acti_mtp, sigma=3)
                            fusion_acti = fusion_acti_mtp_gs/fusion_acti_mtp_gs.max()
                        np.save(fus_acti_path, fusion_acti)
                
               
        
       
#%%
if __name__ == "__main__":
    main()

