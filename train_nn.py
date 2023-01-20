import numpy as np
import h5py as h5
import torch

with h5.File('data/train_data.h5', 'r') as f:
    train_dv    = f['dv'][:]
    train_theta = f['theta'][:]
    dv_fid      = f['dv_fid'][:]
    dv_std      = f['dv_std'][:]
    
with h5.File('data/test_data.h5', 'r') as f:
    test_dv    = f['dv'][:]
    test_theta = f['theta'][:]
    
from emulator import NNEmulator
import torch

N_dim = 5
OUTPUT_DIM = 20

def get_trained_nn_emu(train_cosmo_samples, train_dv_arr):
    """
    Get the trained NN emulator given the training parameters and data vectors
    """
    emu = NNEmulator(N_dim, OUTPUT_DIM, dv_fid, dv_std)
    emu.train(torch.Tensor(train_cosmo_samples), torch.Tensor(train_dv_arr), n_epochs=50)
    return emu

emu = get_trained_nn_emu(train_theta, train_dv)

emu.save('output/model')
