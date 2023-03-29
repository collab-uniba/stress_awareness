import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plot import plot_axes
import constant
import pandas as pd

from hr_preprocessing import calculate_date_time, check_ratio_rr_intervals 

def load_hr():
    '''
    CSV format (does not contain column names):
    ------------------------------------------
                    timestamp   
                        1      
                      value     
                       ...     
                      value     
    ------------------------------------------
    '''

    data = pd.read_csv('hr_explorer\HR.csv', header=None)
    timestamp_0 = data.iloc[0,0]
    data = np.array(data.iloc[2:, 0])

    #Calculate data time form timestamp
    date, times = calculate_date_time(timestamp_0, constant.NUM_HZ_HR, data.shape[0])
   
    return data, date, times



def main():
    #Loading
    data = pd.read_csv('hr_explorer\HR.csv', header=None)
    
    data, date, times = load_hr()
    temp = {} 
    temp['HR'] = data

    plot_axes(temp, date, times)

    check_ratio_rr_intervals(data)


if __name__ == "__main__":
    main()
