import logging

import numpy as np
import pandas as pd

from acc_preprocessing import convert_data_in_g
from acc_filter_test import filters_test

logging.basicConfig(level=logging.DEBUG)





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
    
#if __name__ == "__main__":
#    main()
