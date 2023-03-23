import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import signal
import logging
from skimage.measure import block_reduce
import constant



def convert_data_in_g(data):
    return data / constant.G

def calculate_magnitude_vector(data):
    return np.linalg.norm(data, axis=1)

def aggregating_with_mean(data, hz):
    factor = int(constant.NUM_HZ_ACC / hz)
    pad = factor - (data.shape[0] % factor) # numero di elementi da aggiungere
    data = data.astype('float32')

    if pad != factor:
        data = np.pad(data, (0, pad), constant_values=np.NAN) # aggiunge elementi nan alla fine
    
    data_resampled = np.nanmean(data.reshape(-1, factor), axis=1) # calcola la media ignorando i nan
    
    return data_resampled    
    

def aggregating_with_max(data, hz):
    factor = int(constant.NUM_HZ_ACC / hz)
    pad = factor - (data.shape[0] % factor) # numero di elementi da aggiungere
    data = data.astype('float32')
    
    if pad != factor:
        data = np.pad(data, (0, pad), constant_values=np.NAN) # aggiunge elementi nan alla fine
    
    data_resampled = np.nanmax(data.reshape(-1, factor), axis=1)
   
    return data_resampled      


def aggregating_with_median(data, hz):
    factor = int(constant.NUM_HZ_ACC / hz)
    pad = factor - (data.shape[0] % factor) # numero di elementi da aggiungere
    data = data.astype('float32')
    
    if pad != factor:
        data = np.pad(data, (0, pad), constant_values=np.NAN) # aggiunge elementi nan alla fine
    
    data_resampled = np.nanmedian(data.reshape(-1, factor), axis=1)
   
    return data_resampled    


    