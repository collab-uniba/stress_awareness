import numpy as np

from acc_filtering import HFEN_plus, HFEN, BFEN, ENMO
from acc_preprocessing import calculate_magnitude_vector
from acc_plot import plot_axes

def filters_test(data):
    
    #Filters
    data_en = calculate_magnitude_vector(data)
    data_enmo = ENMO(data)
    data_bfen = BFEN(data)
    data_hfen = HFEN(data)
    data_hfenp = HFEN_plus(data)
    
    metrics_dict = {}
    metrics_dict['EN'] = data_en
    metrics_dict['ENMO'] = data_enmo
    metrics_dict['BFEN'] = data_bfen
    metrics_dict['HFEN'] = data_hfen
    metrics_dict['HFEN_P'] = data_hfenp
    
    plot_axes(metrics_dict)
