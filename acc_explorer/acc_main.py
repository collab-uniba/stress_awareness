import logging

import numpy as np
import pandas as pd

from acc_filtering import HFEN_plus
from acc_preprocessing import convert_data_in_g, aggregating_with_max, aggregating_with_mean
from acc_filter_test import filters_test
import constant

logging.basicConfig(level=logging.DEBUG)


def load_acc():
    '''
    CSV format (does not contain column names):
    ------------------------------------------
    timestamp   |   timestamp   |   timestamp
        32      |       32      |       32
    value_x     |   value_y     |   value_z
    value_x     |   value_y     |   value_z
        ...     |       ...     |       ...
    value_x     |   value_y     |   value_z
    ------------------------------------------
    '''

    data = pd.read_csv('acc_explorer\ACC.csv', header=None)
    
    data = np.array(data)
    
    timestamp_0 = data[0, 0]
    # Remove first (timestamp) and second (Hz) rows
    data = convert_data_in_g(data[2:])

    return data, timestamp_0



def main():
    #Loading
    data, timestamp_0 = load_acc()
    
    filters_test(data, timestamp_0)
    
    #Filtering and removing gravity component
    #data_f = HFEN_plus(data)
    
    #Decrease samples (32Hz => 8Hz)
    #data_f_8 = aggregating_with_max(data_f, 8)

    
    #Releave peak
    #peak_times, peak_values = detects_peak(magn_vector[:10000])
    #peaks_index = detects_peak_window(magn_vector[:10000])
    #print(peaks_index)
    
if __name__ == "__main__":
    main()
