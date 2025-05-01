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

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def main():
    pk_heights = [0.01, 0.1, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875]
    pk_prominence = 0.01 

    for fold_num in np.arange(0, 5):
        # break

        pk_distance = 7
        Beat_FPS = 100 # for madmom

        testset_dirs = {
            'asap':('./datasets/asap/', 
                    './datasets/asap/annotations', 
                    './datasets/asap/activations', 
                    ),


            }
        for pk_height in pk_heights:
            # break

            thre_str = '{:.4f}'.format(pk_height)
            thre_str= thre_str.replace('0.', 'GLB')

            for dataset_name, (main_data_dir, beat_ann_dir, acti_dir) in testset_dirs.items():
                # break
                print("======"*10)
                print("Processing {}".format(dataset_name))
                print("======"*10)
                

                acti_folders = glob.glob(os.path.join(acti_dir, "ABT_f{}".format(fold_num)))

                for acti_folder in acti_folders:
                    acti_type = os.path.basename(acti_folder)
                    print('------'*10)
                    print('Model:{}'.format(acti_type))
                    print('------'*10)
                    # break
                
                    ##### Create folder to save estimations
                    est_foldername = acti_type + '-{}'.format(thre_str) 
                    estimation_dir = os.path.join(main_data_dir, "beat-estimations", est_foldername)
                    if not os.path.exists(estimation_dir):
                        print("---"*20)
                        print("Creating Estimation Folder: {}".format(est_foldername))
                        print("---"*20)
                        Path(estimation_dir).mkdir(parents = True, exist_ok = True)
                    
                    acti_paths = glob.glob(os.path.join(acti_folder, "*.npy"))
                    for acti_path in tqdm.tqdm(acti_paths):
                        # break
                        est_path = os.path.join(estimation_dir, os.path.basename(acti_path).replace(".npy", '.beats'))
                        if not os.path.exists(est_path):
                            beat_acti = np.load(acti_path)[:, 2] #0: non-beat, 1: downbeat, 2: beat
                            beat_acti = gaussian_filter1d(beat_acti, sigma=3)
                            beat_acti = beat_acti/beat_acti.max()
                           

                            beats_pp_tmp, _ = find_peaks(beat_acti, height = pk_height, 
                                           distance = pk_distance, 
                                           prominence = pk_prominence)
                            est = beats_pp_tmp/Beat_FPS

                            np.savetxt(est_path, est, fmt = '%.5f')
    
   
#%%
if __name__ == "__main__":
    main()