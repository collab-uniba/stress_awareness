import numpy as np
import constant
import logging




def remove_gravity_in_z(data):
    data[:, 2] = data[:, 2] - constant.G
    return data

def convert_data_in_g(data):
    return data / constant.G

def calculate_magnitude_vector(data):
    return np.linalg.norm(data, axis=1)