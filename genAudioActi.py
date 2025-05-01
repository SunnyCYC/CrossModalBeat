# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:08:52 2023

@author: sunnycyc
"""

#%%
import os
# from train_modules.dbeatdataset import AudioBeatTrack, AudioBeatDataset
from train_modules import utils
from models.BLSTM_BDB_Model import RNNDownBeatProc
from scipy.special import softmax
import numpy as np
from pathlib import Path
import tqdm

import torch
import json
import glob

dataset_dirs = {

    'asap':('./datasets/asap/', 
              ('./datasets/asap/audio/', '.wav'), 
              './datasets/asap/annotations/'),
    
    }

#%%
def main():

    model_main_dir = './experiments/pretrained_ABTs/'
    model_folders = glob.glob(os.path.join(model_main_dir,"*"))
    
    # cuda_num = int(sys.argv[1])
    cuda_num = 0
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    

    for model_folder in model_folders:
        # break
        #######################################################################
        # load model and related settings
        #######################################################################
        print('===*20')
        print('Processing Model: {}'.format(model_folder))
        print('===*20')
        with open(os.path.join(model_folder,'RNNbeat.json')) as json_file:
            json_data = json.load(json_file)

        model = RNNDownBeatProc(**json_data['model_params'])
        modelpath = os.path.join(model_folder, 'RNNDBeatProc.pth')
        state = torch.load(modelpath, map_location = device)
        model.load_state_dict(state)
        model.cuda(device.index)
        model.eval()
             
        for dname, (dataset_dir, (audio_dir, f_end), ann_dir) in dataset_dirs.items():
            # print(dname, '\n', dataset_dir, '\n', audio_dir, f_end, '\n', ann_dir,'\n' )
            print("---"*20)
            print("Processing {} dataset...".format(dname))
            print("---"*20)
            # break
            #######################################################################
            ### collect audiopaths based on audio_files.txt in each audio_dir
            #######################################################################
            test_txt = os.path.join(audio_dir, 'audio_files.txt')
            test_audiopaths = utils.getAudioPaths(test_txt)
            
            #######################################################################
            ### setup folder directory for beat features based on json information
            #######################################################################
            audio_rate = json_data['exp_setting']['sampling-rate']
            feature_hopsize = json_data['exp_setting']['feature-hopsize']
            # feature_location = os.path.join(dataset_dir, 'beat-features')
            feature_location = os.path.join(dataset_dir,  json_data['exp_setting']['feature-folder'])
            feature_folder = json_data['exp_setting']['feature-type'] + \
                '_SR' + str(audio_rate)+'_HOP'+ str(feature_hopsize)
            feature_dir = os.path.join(feature_location, feature_folder)
            # assert os.path.exists(feature_dir), print('Feature folder not exist: {}'.format(feature_dir))
            
            test_abts = utils.getABTtracks(test_audiopaths, feature_dir, 
                                           ann_dir, audio_rate, feature_hopsize)

            #######################################################################
            # process track-by-track
            #######################################################################
            # model_simpname = json_data['model_info']['model_simpname']
            model_simpname = 'ABT_{}'.format(os.path.basename(model_folder).replace('fold', 'f'))

            acti_sfolder = os.path.join(dataset_dir, 'activations', model_simpname)
            if not os.path.exists(acti_sfolder):
                print('---> Create activation folder:{}'.format(acti_sfolder))
                Path(acti_sfolder).mkdir(parents = True, exist_ok = True)
                
            for abt in tqdm.tqdm(test_abts):
                # break
                actipath = os.path.join(acti_sfolder, os.path.basename(abt.featurepath))
                if not os.path.exists(actipath):
                    feat, beat = abt.get_data()
                    # print(feat.shape, beat.shape)
                    input_feature = torch.tensor(feat[np.newaxis, :, :]).float().to(device)
                    with torch.no_grad():
                        activation = model(input_feature)
                    acti = activation.detach().cpu().numpy()[0, :, :]
                    acti_soft = softmax(acti, axis = 1)
                    # print(acti.shape)
                    np.save(actipath, acti_soft)  ### non-beat, beat
                else:
                    print('---> acti exists:{}'.format(actipath))
#%%
if __name__ == '__main__':
    main()
