import os
import pickle
import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



    
def pickle_data(save_path, data):
    '''
    Save path should end in .pkl or something that is allowed. It should NOT exist
    '''
    with open(save_path, 'wb') as file:
        pickle.dump(data, file)
        

def unpickle_data(save_path):
    with open(save_path, 'rb') as file:
        data = pickle.load(file)
    return data


class HolsteinDataset(Dataset):
    def __init__(self, data:List[Tuple[np.array]], label_step_count:int=1, multi_step_labels:bool=False, n_sims:int=None, sample_random_subset:bool=False, n_steps:int=None):
        '''
        data List[Tuple[np.array]]: Data is an iterable of n_simulations simulations. Each simulation is a tuple of 3 numpy arrays: a 3d complex64 density_mat array, a 2d float32 displacement array, and a 2d float32 momentum array. Each of these 3 arrays has dim 0 as batch dimension, and there are n_steps+1 batches

        label_step_count (int): How many steps after the data point the label should be. For example, if set to 1, the label will be the next time step. Corresponds to n_step_prediction in train.py

        multi_step_labels (bool): Whether to include all intermediate time steps in the label. For example, if not included, the shape of a return rho label (before batching) would be [L,L]. If enabled, it would instead be [label_step_count, L, L]. Corresponds to return_multiple_steps in HolsteinStepSeparate forward method

        n_sims (int | NoneType): The number of simulations to use from the data. If None, will use all simulations available in data. Otherwise, will only use a subset of the data as specified by n_sims

        sample_random_subset (bool): Use in tandem with n_sims specifying a subset of simulations to use. If True, the subset will be a random subset. If False, subset will be the first n_sims simulations

        n_steps (int | NoneType): When enabled, only uses the first n_steps steps from each simulation. Ie. acts as if each simulation only has n_steps+1 data points. If None uses all available steps
        '''

        "Argument checks"
        if label_step_count < 1:
            raise ValueError(f"HolsteinDataset: label_step_count must be at least one. Received: {label_step_count}")
        # If n_sims was specified, make sure it was not greater than the total number of simulations.
        if n_sims is not None:
            if n_sims > len(data):
                raise ValueError(f"HolsteinDataset: n_sims was larger than number of simulations in data! n_sims: {n_sims}\tlen(data): {len(data)}")
        else:
            n_sims = len(data)

        " Data checks "
        # Verify size of first tuple is 3 (others assumed to be same)
        if len(data[0]) != 3:
            raise ValueError(f"HolsteinDataset: Length of first simulation tuple not 3! It was: {len(data[0])}")
        density_mat, displacement, momentum = data[0]
        # Verify dimensionality of each array in first tuple (others assumed to be same)
        if (len(density_mat.shape), len(displacement.shape), len(momentum.shape)) != (3,2,2):
            raise ValueError(f"HolsteinDataset: Dimensionality of first tuple arrays was not 3,2,2. They were: {len(density_mat.shape)},{len(displacement.shape)},{len(momentum.shape)}")
        # Verify dtypes across first simulation data (others assumed to be same)
        if density_mat.dtype != np.complex64 or displacement.dtype != np.float32 or momentum.dtype != np.float32:
            raise ValueError(f"HolsteinDataset: Dtype irregularity! Dtypes:\n\tdensity mat: {density_mat.dtype}\n\tdisplacement: {displacement.dtype}\n\tmomentum: {momentum.dtype}")
        # Verify consistency of system size across first simulation data (others assumed to be same)
        if not (density_mat.shape[1] == density_mat.shape[2] == displacement.shape[1] == momentum.shape[1]):
            raise ValueError(f"HolsteinDataset: Matrix/vector size irregularity! Expected L: {L}. Got shapes:\n\tdensity mat: {density_mat.shape}\n\tdisplacement: {displacement.shape}\n\tmomentum: {momentum.shape}")
        
        " Process the simulations according to arugments "

        # Restrict number of simulations to n_sims
        if not sample_random_subset:
            data = data[:n_sims]
        else:
            data = random.sample(data, n_sims)

        # Find the least number of steps among all simulations
        all_step_counts = [len(sim[0])-1 for sim in data]
        least_step_count = min(all_step_counts)

        if n_steps is not None:
            # if n_steps was specified, verify it was not greater than least_step_count
            if n_steps > least_step_count:
                raise ValueError(f"HolsteinDataset: Cannot specify n_steps which is greater than the least number of steps in any simulation in the data! n_steps: {n_steps}\tleast number of steps in any simulation: {least_step_count}")
            # Then, go through all data points and reduce the length of each simulation to n_steps+1
            data = [tuple(array[:n_steps+1] for array in sim) for sim in data]
        else:
            n_steps = least_step_count

        self.data = data
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.label_step_count = label_step_count
        self.multi_step_labels = multi_step_labels

        # The actual number of data points we can use from each simulation
        # For example, if a simulation has n_steps, then it has n_step+1 data data_points. However, we can only get at most n_steps data points from this data given we do next-step-prediction
        # If our prediction target is greater than next-step, then we get even fewer data points from this data
        self.data_points_per_sim = self.n_steps - (self.label_step_count - 1)

    def __len__(self):
        return self.n_sims * self.data_points_per_sim

    def __getitem__(self, idx):

        ## Todo: Add some comments here describing output shapes

        simulation_idx = idx // self.data_points_per_sim
        step_idx = idx % self.data_points_per_sim

        simulation = self.data[simulation_idx]
        all_density_mat, all_displacement, all_momentum = simulation

        input = torch.tensor(all_density_mat[step_idx], dtype=torch.complex64), torch.tensor(all_displacement[step_idx], dtype=torch.float32), torch.tensor(all_momentum[step_idx], dtype=torch.float32)

        if not self.multi_step_labels:
            rho_label = torch.tensor(all_density_mat[step_idx+self.label_step_count], dtype=torch.complex64)
            Q_label = torch.tensor(all_displacement[step_idx+self.label_step_count], dtype=torch.float32)
            P_label = torch.tensor(all_momentum[step_idx+self.label_step_count], dtype=torch.float32)
        else:
            rho_label = torch.tensor(all_density_mat[step_idx+1:step_idx+self.label_step_count+1], dtype=torch.complex64)
            Q_label = torch.tensor(all_displacement[step_idx+1:step_idx+self.label_step_count+1], dtype=torch.float32)
            P_label = torch.tensor(all_momentum[step_idx+1:step_idx+self.label_step_count+1], dtype=torch.float32)

        label = rho_label, Q_label, P_label

        return input, label


class Virtual_Epoch_loader():

    '''
    A wrapper for a single dataloader. Given a dataloader, define a new number of batches per epoch when the dataloder is iterated. If the dataloader has more batches per epoch than this number, than one actual epoch will span multiple "virtual" epochs. If the data loader has less, then one virtual epoch may have multiple whole or partial actual epochs.

    When a virtual epoch ends and a new one begins, the new one begins at the batch where the previous virtual epoch left off.
    
    '''

    def __init__(self, dataloader:DataLoader, batches_per_epoch:int):
        self.dataloader = dataloader
        self.batches_per_epoch = batches_per_epoch

        self.iterable = iter(self.dataloader)
    
    def __iter__(self):
        self.batches_left = self.batches_per_epoch
        return self

    def __next__(self):
        if self.batches_left > 0:
            self.batches_left -= 1
            try:
                return next(self.iterable)
            except StopIteration:
                self.iterable = iter(self.dataloader)
                return next(self.iterable)
        else:
            raise StopIteration