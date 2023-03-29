import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from scipy import signal
from acc_preprocessing import calculate_magnitude_vector
import constant

#Source of the filters https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3634007/#pone.0061691.s001

def median_filter(data, f_size):
	f_data = signal.medfilt(data, f_size)
	return f_data


def low_pass_filter(data, N, freq):
	num_rows, num_cols=data.shape
	f_data = np.zeros([num_rows, num_cols])
	B,A = signal.butter(N, freq,'low')
	for i in range(num_cols):
		f_data[:, i] = signal.filtfilt(B,A, data[:, i])
	return f_data


def high_pass_filter(data, N, freq):
	num_rows, num_cols=data.shape
	f_data = np.zeros([num_rows, num_cols])
	B,A = signal.butter(N, freq,'high')
	for i in range(num_cols):
		f_data[:, i] = signal.filtfilt(B,A, data[:, i])
	return f_data


def band_pass_filter(data, N, freq1, freq2):
	num_rows, num_cols = data.shape
	f_data = np.zeros([num_rows, num_cols])
	B,A = signal.butter(N, [freq1, freq2],'bandpass', fs=constant.NUM_HZ_ACC)
	for i in range(num_cols):
		f_data[:, i] = signal.filtfilt(B,A, data[:, i])
	
	return f_data
	

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
	freq = 0.1

	data_low_f = low_pass_filter(data, N, freq)
	magn_vec_low_f = calculate_magnitude_vector(data_low_f)

	data_HFEN_plus_f = data_HFEN_f + magn_vec_low_f - 1#G
	#Get max between 0 and value
	data_HFEN_plus_f_p = np.maximum(data_HFEN_plus_f, 0)
	
	return data_HFEN_plus_f_p


def HFEN(data):
	N = 4
	freq = 0.1
	f_data = high_pass_filter(data, N, freq)
	magn_vect = calculate_magnitude_vector(f_data)
	return magn_vect


def BFEN(data):
	N = 4
	freq1 = 0.2
	freq2 = 15
	f_data = band_pass_filter(data, N, freq1, freq2)
	magn_vect = calculate_magnitude_vector(f_data)
	return magn_vect

 
def empatica_filter(data):
#https://support.empatica.com/hc/en-us/articles/202028739-How-is-the-acceleration-data-formatted-in-E4-connect-
#https://stackoverflow.com/questions/43309260/calculate-movement-from-accelerometer-every-second
	num_rows = data.shape[0]
	window_size = constant.NUM_HZ_ACC
	avgs = [0]
	
	num_elem_window = 0
	sum_window = 0

	alpha = 0.9
	#si parte dal secondo elemento perché nel ciclo si considera il precedente 
	for i in range(1, num_rows):
		num_elem_window += 1
		sum_window += max(abs(data[i,0] - data[i-1,0]),
						  abs(data[i,1] - data[i-1,1]), 
						  abs(data[i,2] - data[i-1,2]))

		
		#la seconda condizione è necessaria per l'ultima finestra che potrebbe avere una dimensione minore di window_size
		if (num_elem_window == window_size) or (i == num_rows-1):
			avg = avgs[-1]*alpha + (sum_window/window_size)*(1-alpha)
			avgs.append(avg)

			num_elem_window = 0
			sum_window = 0
				
	return avgs




def android_filter(data):
	#https://developer.android.com/guide/topics/sensors/sensors_motion#sensors-motion-accel
	num_rows = data.shape[0]
	g_x = [0]
	g_y = [0]
	g_z = [0]

	alpha = 0.9
	for i in range(1, num_rows):
		#Isolate the force of gravity with the low-pass filter
		g_x.append(alpha*g_x[-1] + (1 - alpha) * data[i,0])
		g_y.append(alpha*g_y[-1] + (1 - alpha) * data[i,1])
		g_z.append(alpha*g_z[-1] + (1 - alpha) * data[i,2])

	#Remove the gravity contribution
	acc_x = data[:, 0] - np.array(g_x)
	acc_y = data[:, 1] - np.array(g_y)
	acc_z = data[:, 2] - np.array(g_z)

	return acc_x, acc_y, acc_z
