import numpy as np
import pandas as pd
import logging


from acc_preprocessing import calculate_magnitude_vector, convert_data_in_g, remove_gravity_in_z

logging.basicConfig(level=logging.DEBUG)

def extract_data(acc_data):
    data = np.array(acc_data)
    timestamp_0 = data[0, 0]

    # Remove first (timestamp) and second (Hz) rows
    data = data[2:]
    # Remove gravity component from Z-Axis
    data = remove_gravity_in_z(data)

    return timestamp_0, data


def load_acc():
    '''
    CSV format (does not contain column names):
    ------------------------------------------
    timestamp   |   timestamp   |   timestamp
        32      |       32      |       32
    value_x     |   value_y     |   value_z
        ...     |       ...     |       ...
        ...     |       ...     |       ...
    ------------------------------------------
    '''

    data = pd.read_csv('acc_explorer\ACC.csv')

    return data



def main():
    #Loading
    data_raw = load_acc()
    
    #Preprocessing
    timestamp_0, data = extract_data(data_raw)
    
    data_g = convert_data_in_g(data)
    logging.debug(data_g)
    magn_vector = calculate_magnitude_vector(data_g)
    logging.debug(magn_vector)
    #Filtering

if __name__ == "__main__":
    main()
