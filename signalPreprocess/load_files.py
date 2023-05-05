import pandas as pd
import scipy.signal as scisig
import os
import numpy as np


def get_user_input(prompt):
    try:
        return raw_input(prompt)
    except NameError:
        return input(prompt)


def getInputLoadFile(eda, acc, temp):

    data = loadData_E4(eda, acc, temp)
    return data


def _loadSingleFile_E4(data, list_of_columns, expected_sample_rate, freq):

    # Get the startTime and sample rate
    startTime = pd.to_datetime(float(data.columns.values[0]), unit="s")
    sampleRate = float(data.iloc[0][0])
    data = data[data.index != 0]
    data.index = data.index - 1

    # Reset the data frame assuming expected_sample_rate
    data.columns = list_of_columns
    if sampleRate != expected_sample_rate:
        print('ERROR, NOT SAMPLED AT {0}HZ. PROBLEMS WILL OCCUR\n'.format(expected_sample_rate))

    # Make sure data has a sample rate of 8Hz

    data = interpolateDataTo8Hz(data, sampleRate, startTime)

    return data


def loadData_E4(eda_file, acc_file, temperature_file):
    # Load EDA data
    eda_data = _loadSingleFile_E4(eda_file, ["EDA"], 4, "250L")
    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)

    eda_data['filtered_eda'] = butter_lowpass_filter(eda_data['EDA'], 1.0, 8, 6)

    # Load ACC data
    acc_data = _loadSingleFile_E4(acc_file, ["AccelX", "AccelY", "AccelZ"], 32, "31250U")
    # Scale the accelometer to +-2g
    acc_data[["AccelX", "AccelY", "AccelZ"]] = acc_data[["AccelX", "AccelY", "AccelZ"]] / 64.0

    # Load Temperature data
    temperature_data = _loadSingleFile_E4(temperature_file, ["Temp"], 4, "250L")

    data = eda_data.join(acc_data, how='outer')
    data = data.join(temperature_data, how='outer')

    # E4 sometimes records different length files - adjust as necessary
    min_length = min(len(acc_data), len(eda_data), len(temperature_data))

    return data[:min_length]




def loadData_getColNames(data_columns):
    print("Here are the data columns of your file: ")
    print(data_columns)

    # Find the column names for each of the 5 data streams
    colnames = ['EDA data', 'Temperature data', 'Acceleration X', 'Acceleration Y', 'Acceleration Z']
    new_colnames = ['', '', '', '', '']

    for i in range(len(new_colnames)):
        new_colnames[i] = get_user_input("Column name that contains " + colnames[i] + ": ")
        while (new_colnames[i] not in data_columns):
            print("Column not found. Please try again")
            print("Here are the data columns of your file: ")
            print(data_columns)

            new_colnames[i] = get_user_input("Column name that contains " + colnames[i] + ": ")

    # Get user input on sample rate
    sampleRate = get_user_input("Enter sample rate (must be an integer power of 2): ")
    while (sampleRate.isdigit() == False) or (
            np.log(int(sampleRate)) / np.log(2) != np.floor(np.log(int(sampleRate)) / np.log(2))):
        print("Not an integer power of two")
        sampleRate = get_user_input("Enter sample rate (must be a integer power of 2): ")
    sampleRate = int(sampleRate)

    # Get user input on start time
    startTime = pd.to_datetime(get_user_input("Enter a start time (format: YYYY-MM-DD HH:MM:SS): "))
    while type(startTime) == str:
        print("Not a valid date/time")
        startTime = pd.to_datetime(get_user_input("Enter a start time (format: YYYY-MM-DD HH:MM:SS): "))

    return sampleRate, startTime, new_colnames


def interpolateDataTo8Hz(data, sample_rate, startTime):
    if sample_rate < 8:
        # Upsample by linear interpolation
        if sample_rate == 2:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='500L')
        elif sample_rate == 4:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='250L')
        data = data.resample("125L").mean()
    else:
        if sample_rate > 8:
            # Downsample
            idx_range = list(range(0, len(data)))  # TODO: double check this one
            data = data.iloc[idx_range[0::int(int(sample_rate) / 8)]]
        # Set the index to be 8Hz
        data.index = pd.date_range(start=startTime, periods=len(data), freq='125L')

    # Interpolate all empty values
    data = interpolateEmptyValues(data)
    return data


def interpolateEmptyValues(data):
    cols = data.columns.values
    for c in cols:
        data.loc[:, c] = data[c].interpolate()

    return data


def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y