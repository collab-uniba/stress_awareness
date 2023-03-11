import logging

import numpy as np
import pandas as pd

from acc_filtering import HFEN_plus
from acc_preprocessing import convert_data_in_g
from acc_filter_test import filters_test

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

    data = pd.read_csv('acc_explorer\ACC.csv')
    data = np.array(data)
    
    timestamp_0 = data[0, 0]
    
    # Remove first (timestamp) and second (Hz) rows
    data = convert_data_in_g(data[2:])
    
    return timestamp_0, data



def main():
    #Loading
    timestamp_0, data = load_acc()
    
    filters_test(data)

    #Filtering
    data_f = HFEN_plus(data)
    
    
    
    
    
    

    
if __name__ == "__main__":
    main()
