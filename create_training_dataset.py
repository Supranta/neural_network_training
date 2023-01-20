import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
import h5py as h5

def get_Pk(cosmo_pars):
    As, ns, H0, ombh2, omch2 = cosmo_pars
    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(As=As, ns=ns)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0.], kmax=2.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-3, maxkh=1, npoints = 2000)
    s8 = np.array(results.get_sigma8())
    
    return kh, pk[0], s8


# We use 21 logarithmically spaced k bins between 0.01 h/Mpc to 0.1 h/Mpc for our inference

k_min = 0.01
k_max = 0.1
N_bins = 21

k_bins = np.logspace(np.log10(k_min), np.log10(k_max), N_bins)

# Functions to get the binned linear power spectra.

from scipy.interpolate import interp1d
from scipy.integrate import quad

EPS_REL = 1e-4

def integration_numerator(u, pk_interp):
    kh = np.exp(u)
    jacobian = kh
    return pk_interp(kh) * kh * kh * jacobian

def integration_denominator(u):
    kh = np.exp(u)
    jacobian = kh
    return kh * kh * jacobian

def integration_k_bincentre(u):
    kh = np.exp(u)
    jacobian = kh
    return kh * kh * kh * jacobian

def get_binned_Pk(kh, pk, k_bins):
    pk_interp = interp1d(kh, pk)
    binned_Pk_list = []
    binned_k_list = []
    for i in range(len(k_bins) - 1):
        u_lo = np.log(k_bins[i])
        u_hi = np.log(k_bins[i+1])
        I_n, _ = quad(integration_numerator, u_lo, u_hi, epsrel=EPS_REL, args=(pk_interp,))
        I_d, _ = quad(integration_denominator, u_lo, u_hi, epsrel=EPS_REL)
        I_k, _ = quad(integration_k_bincentre, u_lo, u_hi, epsrel=EPS_REL)
        
        Pk_i = I_n / I_d
        k_i  = I_k / I_d
        
        binned_Pk_list.append(Pk_i)
        binned_k_list.append(k_i)
        
    return np.array(binned_k_list), np.array(binned_Pk_list)

def compute_datavector(cosmo_pars):
    kh, pk, _ = get_Pk(cosmo_pars)
    _, Pk_binned = get_binned_Pk(kh, pk, k_bins)
    return Pk_binned

# We use the following fiducial parameters in our analysis

cosmo_pars_fid = np.array([2e-9, 0.97, 70., 0.0228528, 0.1199772])

kh_fid, pk_fid, _ = get_Pk(cosmo_pars_fid)
binned_k, binned_Pk_fid = get_binned_Pk(kh_fid, pk_fid, k_bins)

delta_k = (k_bins[1:] - k_bins[:-1])
# We assume a volume of NGC-High-z chunk (eqn A8 of https://arxiv.org/pdf/2009.00622.pdf)
V = 2.78 * 10**9      
Pk_cov = 4 * np.pi**2 * binned_Pk_fid**2 / binned_k**2 / delta_k / V

dv_fid = compute_datavector(cosmo_pars_fid)
dv_std = np.sqrt(Pk_cov)

from pyDOE import lhs

N_dim = 5
cosmo_prior = np.array([[1.7e-9, 2.5e-9],
                       [0.91, 1.01],
                       [61, 73],
                       [0.014, 0.035],
                       [0.06, 0.2]])

def get_cosmo_lhs_samples(N_samples, cosmo_prior):
    lhs_samples = lhs(N_dim, N_samples)
    cosmo_samples = cosmo_prior[:,0] + (cosmo_prior[:,1] - cosmo_prior[:,0]) * lhs_samples
    return cosmo_samples

N_lhs_samples = 2000

train_cosmo_samples = get_cosmo_lhs_samples(N_lhs_samples, cosmo_prior)
test_cosmo_samples  = get_cosmo_lhs_samples(N_lhs_samples // 10, cosmo_prior)

from tqdm import tqdm
from multiprocessing import Pool

def calculate_datavector_batch(train_cosmo_samples):
    """
    Function to calculate the data vectors for a batch of training samples
    """
    train_dv_list = []
    with Pool() as p:
        train_dv_list = list(tqdm(p.imap(compute_datavector, train_cosmo_samples), total=len(train_cosmo_samples)))
    return np.array(train_dv_list)    

train_dv_arr = calculate_datavector_batch(train_cosmo_samples)
test_dv_arr = calculate_datavector_batch(test_cosmo_samples)

def save_data(filename, dv_arr, cosmo_samples):
    with h5.File(filename, 'w') as f:
        f['dv']     = dv_arr
        f['theta']  = cosmo_samples
        f['dv_fid'] = dv_fid 
        f['dv_std'] = dv_std 
        
save_data('train_data.h5', train_dv_arr, train_cosmo_samples)        
save_data('test_data.h5', test_dv_arr, test_cosmo_samples)        