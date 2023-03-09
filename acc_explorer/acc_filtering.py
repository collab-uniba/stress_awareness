import numpy as np
from scipy import signal


def median_filter(data, f_size):
	num_row, num_col=data.shape
	f_data=np.zeros([num_row, num_col])
	for i in range(num_col):
		f_data[:,i]=signal.medfilt(data[:,i], f_size)
	return f_data


'''
def freq_filter(data, f_size, cutoff):
	num_row, num_col=data.shape
	f_data=np.zeros([num_row, num_col])
	lpf=signal.firwin(f_size, cutoff)
	for i in range(num_col):
		f_data[:,i]=signal.convolve(data[:,i], lpf, mode='same')
	return f_data

def median_and_freq_filter(data, f_size, cutoff):
	f_med_data = median_filter(data, f_size)
	f_med_freq_data = freq_filter(f_med_data, f_size, cutoff)
	return f_med_freq_data

def freq_and_median_filter(data, f_size, cutoff):
	f_freq_data = freq_filter(data, f_size, cutoff)
	f_freq_med_data = median_filter(f_freq_data, f_size)
	return f_freq_med_data
'''