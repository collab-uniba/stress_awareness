import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constant
import numpy as np
from acc_preprocessing import aggregating_with_mean, aggregating_with_max, aggregating_with_median
from plot import plot_axes
from acc_filtering import HFEN_plus, HFEN, BFEN, ENMO, empatica_filter, android_filter
from acc_preprocessing import calculate_magnitude_vector

from hr_explorer.hr_preprocessing import calculate_date_time


def filters_test(data, timestamp_0):
    
     #Filters
    '''
    data_en = calculate_magnitude_vector(data)
    
    data_bfen = BFEN(data)
    data_hfen = HFEN(data)
    '''
    data_hfenp = HFEN_plus(data)
    data_enmo = ENMO(data)
    x, y, z = android_filter(data)
    data_android_filter_matrix = np.stack((x, y, z), axis=-1)
    data_android_filter = calculate_magnitude_vector(data_android_filter_matrix)
    #Calculate data and times from timestamp
    date, times = calculate_date_time(timestamp_0, constant.NUM_HZ_ACC, x.shape[0])
    
    metrics_dict = {}
    metrics_dict['xg'] = data[:,0]
    metrics_dict['yg'] = data[:,1]
    metrics_dict['zg'] = data[:,2]
    metrics_dict['x'] = x
    metrics_dict['y'] = y
    metrics_dict['z'] = z
    metrics_dict['android'] = data_android_filter
    metrics_dict['HFEN_P'] = data_hfenp
    metrics_dict['ENMO'] = data_enmo
    '''
    metrics_dict['EN'] = data_en
    
    metrics_dict['BFEN'] = data_bfen
    metrics_dict['HFEN'] = data_hfen
    '''
    
    plot_axes(metrics_dict, date, times)
    

    metrics_dict = {}
    data_empa = empatica_filter(data)
    data_android_empa = empatica_filter(data_android_filter_matrix)
    date, times = calculate_date_time(timestamp_0, 1, len(data_empa))
    
    metrics_dict['EMPATICA FILT'] = data_empa
    metrics_dict['ANDROID EMPA'] = data_android_empa
    plot_axes(metrics_dict, date, times)
   
   
   
    '''
    data_f_8_mean = aggregating_with_mean(data_hfenp, 8) 
    data_f_8_max = aggregating_with_max(data_hfenp, 8)
    data_f_8_median = aggregating_with_median(data_hfenp, 8)
    

    metric_dict1 = {}
    metric_dict1['8 Hz mean'] = data_f_8_mean
    metric_dict1['8 Hz max'] = data_f_8_max
    metric_dict1['8 Hz median'] = data_f_8_median
    
    #Calculate data and times from timestamp
    date, times = calculate_date_time(timestamp_0, 8, data_f_8_mean.shape[0])
    
    plot_axes(metric_dict1, date, times)
    '''