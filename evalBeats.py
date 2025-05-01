# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:07:19 2023


@author: Sunny
"""

#%%

import os
from PredominantLocalPulseModules import tempo_modules as TempModules
import tqdm
import glob
import pandas as pd
import numpy as np
import mir_eval
from pathlib import Path


f_measure_threshold = 0.07

def renameLc(lc_res, lvalue):
    rename_lc_res = {}
    for k, v, in lc_res.items():
        # break
        rename_k = k.replace('-', '-L{}-'.format(lvalue))
        rename_lc_res[rename_k] = v
    return rename_lc_res


def main():
    ## specify dataset name, estimation dir, and downbeat annotation location
    csv_out_maindir = './evaluation_results/'
    csv_out_folder = os.path.join(csv_out_maindir, "csvfiles")
    
    if not os.path.exists(csv_out_folder):
        print("==="*20)
        print('Creating folder for csv files :{}'.format(csv_out_folder))
        # print('Method:{}'.format(exp_foldername))
        print("==="*20)
        Path(csv_out_folder).mkdir(parents = True, exist_ok = True)
    Lvalues = [2, 3, 4]
    ### set target dataset dirs : name:(main_dir, beat_ann_dir, dbeat_ann_dir,  acti_dir)
    est_dicts = {
        'asap':('./datasets/asap/', 
                './datasets/asap/annotations', 
                './datasets/asap/activations', 
                  ),
        }
    
    #######################################################
    # process for each dataset
    #######################################################
    for dataset_name, (main_data_dir, beat_ann_dir, acti_dir) in est_dicts.items():
        # break
        exp_foldernames = os.listdir(os.path.join(main_data_dir, 'beat-estimations'))

        for exp_foldername in exp_foldernames:
            # break
            songlevel_results = []

            est_folder = os.path.join(main_data_dir, 'beat-estimations', exp_foldername)
            est_method = os.path.basename(est_folder)
            ##########################################################
            # process all estimation method for each dataset
            ##########################################################

            print("==="*27)
            print("Evaluation for Dataset: {}, Estimation: {}".format(dataset_name, est_method))
            print("==="*27)

            ##########################################################
            # process all tracks for each estimation method
            ##########################################################
            est_files = glob.glob(os.path.join(est_folder, "*.beats"))
            # est_files = glob.glob(os.path.join(est_dir, est_method, "*.beats"))
            for est_file in tqdm.tqdm(est_files):
                ###### get beat annotations for ACR
                beat_annpath = os.path.join(beat_ann_dir, os.path.basename(est_file))

                beat_ann = np.loadtxt(beat_annpath)
                if len(beat_ann.shape)==2:
                    beat_ann = beat_ann[:, 0]
                ###### calculate statistics

                ibi_stability = TempModules.tempoStability(beat_annpath, deviation = 0.04)
                b_cvar = TempModules.calCvar(TempModules.getTempoCurve(beat_annpath))
                tempocurve= TempModules.getTempoCurve(beat_annpath)
                meanTrackTempo = TempModules.getSongMeanTempo(beat_annpath)
                
                ###### exclude the measure number axis
                est = np.loadtxt(est_file)
                if len(est.shape)==2:
                    est = est[:, 0]
                if est.shape==(): # with only one estimation
                    est = np.array([est])

                F, P, R = mir_eval.onset.f_measure(beat_ann, est, window=f_measure_threshold)
                print('F: {:.3f}, P: {:.3f}, R: {:.3f}'.format(F, P, R))
                # Recall, Precision = RnP(beat_ann , est, f_measure_threshold= f_measure_threshold)
                result_dict = {
                    'Dataset': dataset_name, 
                    'Method': est_method,
                    'Setting': est_method.split('-')[1],
                    'Track': est_file, 
                    'F1': F, 
                    'R': R, 
                    'P': P, 
                    'IBI-Stability': ibi_stability, 
                    'B-Cvar': b_cvar, 
                    'Track-Mean-Tempo': meanTrackTempo, 
                    'Track-Max-Tempo': tempocurve.max(), 
                    'Track-Min-Tempo':tempocurve.min(), 
                    }

                for Lvalue in Lvalues:
                    # break
                    lc_res = TempModules.Lcorrect_eval(est, beat_ann, 
                                                  tolerance = f_measure_threshold, 
                                                  L = Lvalue)
                    result_dict.update(renameLc(lc_res, Lvalue))
                songlevel_results.append(result_dict)
            csv_spath = os.path.join(csv_out_folder, "{}-{}_songlev_evalutation.csv".format(dataset_name, exp_foldername))
            df = pd.DataFrame(songlevel_results)
            df.to_csv(csv_spath)


#%%
    
if __name__=="__main__":
    main()