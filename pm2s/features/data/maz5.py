import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../'))

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import LightningDataModule

from data.io import read_note_sequence
from data.constants import tolerance, resolution

data_dir = os.path.join(os.environ['DATADIR'], 'original_datasets/Maz-5')
feat_dir = os.path.join(os.environ['DATADIR'], 'precomputed_features/Maz-5')

# Path definitions
dataset_paths = {
    'midi_path_align': os.path.join(data_dir, 'jzben-align'),   # Aligned MIDI
    'midi_path_kong': os.path.join(data_dir, 'bytedance'),      # Transcribed MIDI using Kong's model
    'midi_path_ben': os.path.join(data_dir, 'ben'),             # Transcribed MIDI using Ben's model
    'annot_path': os.path.join(data_dir, 'annotations_beatMeasure_FiveMazurkas_FD'),
}
feature_paths = {
    'midi_path_align': os.path.join(feat_dir, 'noteseq_align'),
    'midi_path_kong': os.path.join(feat_dir, 'noteseq_kong'),
    'midi_path_ben': os.path.join(feat_dir, 'noteseq_ben'),
    'annot_path_align': os.path.join(feat_dir, 'beat_annot_align'),
    'annot_path_kong': os.path.join(feat_dir, 'beat_annot_kong'),
    'annot_path_ben': os.path.join(feat_dir, 'beat_annot_ben'),
}
for p in feature_paths.values():
    os.makedirs(p, exist_ok=True)
    

class Maz5DataModule(LightningDataModule):


    def __init__(self, 
        midi_version: str, 
        fold_idx: int, 
        max_length: int,
        batch_size_train: int, 
        batch_size_eval: int,
        num_workers: int, 
        input_features: list, 
        print_statistics: bool = False,
    ):

        super().__init__()

        assert midi_version in ['align', 'kong', 'ben', 'all']
        self.midi_version  = midi_version
        if midi_version == 'all':
            self.dataset_paths_x = [dataset_paths['midi_path_align'], dataset_paths['midi_path_kong'], dataset_paths['midi_path_ben']]
            self.dataset_paths_y = [dataset_paths['annot_path']]
            self.feature_paths_x = [feature_paths['midi_path_align'], feature_paths['midi_path_kong'], feature_paths['midi_path_ben']]
            self.feature_paths_y = [feature_paths['annot_path_align'], feature_paths['annot_path_kong'], feature_paths['annot_path_ben']]
        else:
            self.dataset_paths_x = [dataset_paths['midi_path_{}'.format(midi_version)]]
            self.dataset_paths_y = [dataset_paths['annot_path']]
            self.feature_paths_x = [feature_paths['midi_path_{}'.format(midi_version)]]
            self.feature_paths_y = [feature_paths['annot_path_{}'.format(midi_version)]]

        assert fold_idx in range(5)
        self.fold_idx = fold_idx

        self.max_length = max_length
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers

        assert all(feature in ['pitch', 'onset', 'duration', 'velocity'] for feature in input_features)
        self.input_features = input_features

        self.print_statistics = print_statistics


    def setup(self, stage = None):
        ## Dataset splits
        maz5_split_all_folds = json.load(open('../dataset_splits/maz5_split.json', 'r'))
        split = maz5_split_all_folds['fold{}'.format(self.fold_idx)]

        fns_all = [fn.replace('.mid', '.npy') for fn in os.listdir(self.dataset_paths_x[0])]
        self.fns_train  = [fn for fn in fns_all if any(prefix in fn for prefix in split['train'])]
        self.fns_val    = [fn for fn in fns_all if any(prefix in fn for prefix in split['val'])]
        self.fns_test   = [fn for fn in fns_all if any(prefix in fn for prefix in split['test'])]

        count_1s_beat, count_0s_beat, count_1s_downbeat, count_0s_downbeat = 0, 0, 0, 0
        lengths = []

        for fn_idx, fn in enumerate(fns_all):
            print('Preparing fn {}/{}'.format(fn_idx+1, len(fns_all)), end='\r')

            for midi_version_idx in range(len(self.dataset_paths_x)):

                # Note sequence (pitch, onset, duration, velocity)
                if not os.path.exists(os.path.join(self.feature_paths_x[midi_version_idx], fn)):
                    note_seq = read_note_sequence(os.path.join(self.dataset_paths_x[midi_version_idx], fn.replace('.npy', '.mid')))  # (length, 4)
                    np.save(os.path.join(self.feature_paths_x[midi_version_idx], fn), note_seq)
                else:
                    note_seq = np.load(os.path.join(self.feature_paths_x[midi_version_idx], fn))  # (length, 4)

                # Beats and downbeats
                if not os.path.exists(os.path.join(self.feature_paths_y[midi_version_idx], fn)):
                    annot = pd.read_csv(os.path.join(self.dataset_paths_y[0], fn[:15], fn.replace('.npy', '.csv')), header=None)
                    beats = annot[0].to_numpy()
                    downbeats = np.array([annot[0][i] for i in range(len(annot)) if annot[1][i] % 1 == 0])
                    
                    probs_beat = np.array([np.min(np.abs(beats - note_seq[i,1])) < tolerance for i in range(len(note_seq))]).astype(int)
                    probs_downbeat = np.array([np.min(np.abs(downbeats - note_seq[i,1])) < tolerance for i in range(len(note_seq))]).astype(int)
                    ibis = []
                    for i in range(len(note_seq)):
                        l = len((beats - note_seq[i,1]) < 0) - 1
                        if l == -1: l += 1
                        if l+1 == len(beats): l -= 1
                        ibis.append(beats[l+1] - beats[l])
                    
                    outputs = np.array([probs_beat, probs_downbeat, ibis])  # (3, length)
                    np.save(os.path.join(self.feature_paths_y[midi_version_idx], fn), outputs)
                else:
                    outputs = np.load(os.path.join(self.feature_paths_y[midi_version_idx], fn))  # (3, length)

                # Update statistics
                count_1s_beat += np.sum(outputs[0] == 1)
                count_0s_beat += np.sum(outputs[0] == 0)
                count_1s_downbeat += np.sum(outputs[1] == 1)
                count_0s_downbeat += np.sum(outputs[1] == 0)
                lengths.append(len(note_seq))
        
        count_notes = count_1s_beat + count_0s_beat
        
        print()

        # Statistics
        if self.print_statistics:

            print('\t\tones\tzeros')
            print('beat\t\t{:.2f}\t{:.2f}'.format(count_1s_beat / count_notes, count_0s_beat / count_notes))
            print('downbeat\t{:.2f}\t{:.2f}'.format(count_1s_downbeat / count_notes, count_0s_downbeat / count_notes))
            print('Average length (note_seq): {:.2f}'.format(np.mean(lengths)))
            print('Std lengths (note seq): {:.2f}'.format(np.std(lengths)))
            print('Length range (note sequence): ({}, {})'.format(np.min(lengths), np.max(lengths)))

            plt.figure()
            plt.hist(lengths)
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/note_seq_length_dist.pdf')


    def train_dataloader(self):
        dataset = Maz5Dataset(fns=self.fns_train, midi_version=self.midi_version, input_features=self.input_features, max_length=self.max_length, eval=False)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size = self.batch_size_train,
            sampler = sampler,
            num_workers = self.num_workers,
            drop_last = True,
        )
        return dataloader


    def val_dataloader(self):
        dataset = Maz5Dataset(fns=self.fns_val, midi_version=self.midi_version, input_features=self.input_features, max_length=self.max_length, eval=True)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size = self.batch_size_eval,
            sampler = sampler,
            num_workers = self.num_workers,
            drop_last = False,
        )
        return dataloader


    def test_dataloader(self):
        dataset = Maz5Dataset(fns=self.fns_test, midi_version=self.midi_version, input_features=self.input_features, max_length=self.max_length, eval=True)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size = self.batch_size_eval,
            sampler = sampler,
            num_workers = self.num_workers,
            drop_last = False,
        )
        return dataloader


class Maz5Dataset(torch.utils.data.Dataset):

    
    def __init__(self, fns: list, midi_version: str, input_features: list, max_length: int, eval: bool):

        self.fns = fns
        if midi_version == 'all':
            self.feature_paths_x = [feature_paths['midi_path_align'], feature_paths['midi_path_kong'], feature_paths['midi_path_ben']]
            self.feature_paths_y = [feature_paths['annot_path_align'], feature_paths['annot_path_kong'], feature_paths['annot_path_ben']]
        else:
            self.feature_paths_x = [feature_paths['midi_path_{}'.format(midi_version)]]
            self.feature_paths_y = [feature_paths['annot_path_{}'.format(midi_version)]]
            
        self.input_features = input_features
        self.max_length = max_length
        self.eval = eval

        self.len = len(fns) * 2  # We sample self.max_length notes per sequence during training, while the avearge note sequence length for the dataset is 1084.80 notes.
        if midi_version == 'all':
            self.len *= 3   # three MIDI versions.

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Load data
        fn_idx = idx % len(self.fns)
        midi_version_idx = 0 if len(self.feature_paths_x) == 1 else idx // (self.len // 3)

        fn = self.fns[fn_idx]
        x = np.load(os.path.join(self.feature_paths_x[midi_version_idx], fn))
        y = np.load(os.path.join(self.feature_paths_y[midi_version_idx], fn))

        if not self.eval:
            # Data augmentation
            x, y = self.data_augmentation(x, y)

            # Sample segment (self.max_length notes) & input features
            length = len(x)
            start_idx = np.random.choice(range(max(1, len(x) - self.max_length)))   # All performances are longer than self.max_length
            x = x[start_idx: start_idx + self.max_length, :]
            y = y[:, start_idx: start_idx + self.max_length]

            # Padding
            if length < self.max_length:
                x = np.concatenate([x, np.zeros((self.max_length - length, 4))], axis=0)
                y = np.concatenate([y, np.zeros((3, self.max_length - length))], axis=1)

        else:
            length = len(x)
            # Pad all sequences into the maximum length in the dataset (247, 2046)
            x = np.concatenate([x, np.zeros((2500 - length, 4))], axis=0)
            y = np.concatenate([y, np.zeros((3, 2500 - length))], axis=1)
        
        # Use the specified input features
        if 'velocity' not in self.input_features: x[:, 3] = 60
        if 'duration' not in self.input_features: x[:, 2] = 0.1
        

        # Separate y
        probs_beat = y[0,:].astype(float)
        probs_downbeat = y[1,:].astype(float)
        ibis = np.round(np.clip(y[2,:], 0, 4) / resolution).astype(int)  # Convert ibi into categorical

        return x, (probs_beat, probs_downbeat, ibis), length
    
    @staticmethod
    def data_augmentation(x, y):
        # tempo change
        tempo_change_ratio = random.uniform(0.8, 1.2)
        x[:, 1:3] *= 1 / tempo_change_ratio  # onset and duration
        y[2,:] *= 1 / tempo_change_ratio   # ibi

        # pitch shift
        shift = round(random.uniform(-12, 12))
        x[:, 0] += shift

        # extra notes
        x_new = np.zeros((len(x) * 2, 4))  # duplicate
        y_new = np.zeros((3, len(x) * 2))
        x_new[::2,:] = np.copy(x)   # original notes
        y_new[:,::2] = np.copy(y)
        x_new[1::2,:] = np.copy(x)   # extra notes
        y_new[:, 1::2] = np.copy(y)
        # octave shift (+-12) for extra notes only
        octave_shift = ((np.round(np.random.random(len(x_new))) - 0.5) * 24). astype(int)
        octave_shift[::2] = 0
        x_new[:,0] += octave_shift
        x_new[:,0][x_new[:,0] < 0] += 12
        x_new[:,0][x_new[:,0] > 127] -= 12
        # random ratio of extra notes
        ratio = random.random() * 0.3
        probs = np.random.random(len(x_new))
        probs[::2] = 0
        remaining = probs < ratio
        x_new = x_new[remaining,:]
        y_new = y_new[:, remaining]

        # missing notes
        concurrent_notes = np.diff(x[:, 1]) < tolerance
        ratio = random.random()
        p = concurrent_notes * np.random.random(len(concurrent_notes))
        remaining = np.concatenate([np.array([True]), p < (1 - ratio)])
        x = x[remaining,:]
        y = y[:, remaining]

        return x, y
