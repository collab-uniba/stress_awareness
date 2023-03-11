import numpy as np
from scipy import signal
import constant

from acc_preprocessing import calculate_magnitude_vector

#Source of the filters https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3634007/#pone.0061691.s001

def median_filter(data, f_size):
	f_data = signal.medfilt(data, f_size)
	return f_data


def low_pass_filter(data, N, omega):
	num_row, num_col=data.shape
	f_data = np.zeros([num_row, num_col])
	B,A = signal.butter(N, omega,'low')
	for i in range(num_col):
		f_data[:, i] = signal.filtfilt(B,A, data[:, i])
	return f_data


def high_pass_filter(data, N, omega):
	num_row, num_col=data.shape
	f_data = np.zeros([num_row, num_col])
	B,A = signal.butter(N, omega,'high')
	for i in range(num_col):
		f_data[:, i] = signal.filtfilt(B,A, data[:, i])
	return f_data


def band_pass_filter(data, N, omega_0, omega_1):
	num_row, num_col=data.shape
	f_data = np.zeros([num_row, num_col])
	B,A = signal.butter(N, [omega_0, omega_1],'bandpass', fs=constant.NUM_HZ)
	for i in range(num_col):
		f_data[:, i] = signal.filtfilt(B,A, data[:, i])
	
	return f_data
	


'''
def e4_processing(data):
	num_row = data.shape[0]
	avgs = [0]
	for i in range(1, num_row - constant.NUM_HZ):
		sum_window = 0
		for j in range(0, constant.NUM_HZ):
			sum_window += max(abs(data[i+j,0] - data[i+j-1,0]),
			                  abs(data[i+j,1] - data[i+j-1,1]), 
							  abs(data[i+j,2] - data[i+j-1,2]))
			j += 1
		avg = avgs[i-1]*0.9 + (sum_window/constant.NUM_HZ)*0.1
		avgs.append(avg)
	return avgs
'''

def ENMO(data):
	magn_vec_data = calculate_magnitude_vector(data)

	#Subtract 1G
	magn_vec_data -= 1

	#Get max between 0 and value
	magn_vec_data_p = np.maximum(magn_vec_data, 0)

	return magn_vec_data_p


def HFEN_plus(data):
	data_HFEN_f = HFEN(data)

	N = 4
	omega = 0.2

	data_low_f = low_pass_filter(data, N, omega)
	magn_vec_low_f = calculate_magnitude_vector(data_low_f)

	data_HFEN_plus_f = data_HFEN_f + magn_vec_low_f - 1#G
	
	#Get max between 0 and value
	data_HFEN_plus_f_p = np.maximum(data_HFEN_plus_f, 0)

	return data_HFEN_plus_f_p


def HFEN(data):
	N = 4
	omega = 0.2
	f_data = high_pass_filter(data, N, omega)
	magn_vect = calculate_magnitude_vector(f_data)
	return magn_vect


def BFEN(data):
	N = 4
	omega_0 = 0.2
	omega_1 = 15
	f_data = band_pass_filter(data, N, omega_0, omega_1)
	magn_vect = calculate_magnitude_vector(f_data)
	return magn_vect


