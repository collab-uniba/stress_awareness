import pandas as pd
import numpy as np
import os
import scipy.signal as scisig
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
SAMPLE_RATE = 8

def get_seconds_and_microseconds(pandas_time):
    return pandas_time.seconds + pandas_time.microseconds * 1e-6

def _loadSingleFile_E4(filepath, list_of_columns, expected_sample_rate, freq):
    # Load data
    data = pd.read_csv(filepath)

    # Get the startTime and sample rate
    startTime = pd.to_datetime(float(data.columns.values[0]), unit="s")
    sampleRate = float(data.iloc[0][0])
    print("sample rate" + str(sampleRate))
    print("StartTime" + str(startTime))
    data = data[data.index != 0]
    data.index = data.index - 1

    # Reset the data frame assuming expected_sample_rate
    data.columns = list_of_columns
    if sampleRate != expected_sample_rate:
        print('ERROR, NOT SAMPLED AT {0}HZ. PROBLEMS WILL OCCUR\n'.format(expected_sample_rate))


    # Make sure data has a sample rate of 8Hz
    data = interpolateDataTo8Hz(data, sampleRate, startTime)

    return data


def loadData_E4(filepath):
    # Load EDA data
    eda_data = _loadSingleFile_E4(os.path.join(filepath, 'EDA.csv'), ["EDA"], 4, "250L")

    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
    eda_data['filtered_eda'] = butter_lowpass_filter(eda_data['EDA'], 1.0, 8, 6)

    # Load ACC data
    acc_data = _loadSingleFile_E4(os.path.join(filepath, 'ACC.csv'), ["AccelX", "AccelY", "AccelZ"], 32, "31250U")
    # Scale the accelometer to +-2g
    acc_data[["AccelX", "AccelY", "AccelZ"]] = acc_data[["AccelX", "AccelY", "AccelZ"]] / 64.0

    # Load Temperature data
    temperature_data = _loadSingleFile_E4(os.path.join(filepath, 'TEMP.csv'), ["Temp"], 4, "250L")

    data = eda_data.join(acc_data, how='outer')
    data = data.join(temperature_data, how='outer')

    # E4 sometimes records different length files - adjust as necessary
    min_length = min(len(acc_data), len(eda_data), len(temperature_data))

    return data[:min_length]


def interpolateDataTo8Hz(data, sample_rate, startTime):
    if sample_rate<8:
        # Upsample by linear interpolation
        if sample_rate==2:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='500L')
        elif sample_rate == 4:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='250L')
        data = data.resample("125L").mean()
    else:
        if sample_rate>8:
            # Downsample
            idx_range = list(range(0,len(data))) # TODO: double check this one
            data = data.iloc[idx_range[0::int(int(sample_rate)/8)]]
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

def detect_fast_edges(data):


    data['Timestamp'] = data.index
    data = data.reset_index()
    #data['Timestamp'] = data['Timestamp'].astype('datetime64[s]')

    n = data.loc[data['Timestamp'] == '2019-10-31 11:10:48'].index[0]
    print('i' + str(n))
    y = data['filtered_eda'].values
    x = y[n:]
    # condition that had to be satisfy
    # (x(t) - x(t-1))/x(t-1) > 0.1 al secondo
    print(len(data['filtered_eda'].values))
    dx = np.insert(np.diff(x) / np.abs(x[:-1]), 0, 0) # (x(t) - x(t-1))/x(t-1) > 0.1 al secondo
    filt = [np.nan if t > (10/100/8) or t < -(1/100/8) else y for y,t in zip(x,dx)]

    with open(r'C:\Users\user\Desktop\FieldStudy\ASML\P18\week1\10_31_2019, 12_00_38\fast_peak.csv', 'w') as f:
        for item in filt:
            f.write("%s\n" % item)

    data['filtered_eda'] = filt



    data['filtered_eda'] = data['filtered_eda'].ffill()
    #yfilt = pd.Series(filt)
    #yfilt = yfilt.interpolate(method='nearest')
    #yfilt = yfilt.interpolate(method='linear')
    #yfilt = yfilt.to_numpy()
    #data['filtered_eda'] = yfilt




    #moving average filter
    y = np.convolve(x, np.ones(3), 'same') / 3
    print(len(y))
    data['filtered_eda'] = y

    print(data['filtered_eda'])
    with open(r'C:\Users\user\Desktop\FieldStudy\ASML\P18\week1\10_31_2019, 12_00_38\fast_peak2.csv', 'w') as f:
        for item in data['filtered_eda']:
            f.write("%s\n" % item)


    return data



# def moving_average_filter(data):
#
#     y = data['filtered_eda'].values
#     y = np.convolve(y, np.ones(3), 'valid') / 3
#     print(len(y))
#     print(len(data['filtered_eda'].values))
    #data['filtered_eda'] = y

def percentage(part, whole):
  return float(whole) * float(part)/100

def plotPeaks(data, x_seconds=True, sampleRate=8):
    list_time = []
    fs = 8
    # for y in data.index.values:
    #     print(y)
    #     d = datetime.strptime(str(y), '%Y-%m-%dT%H:%M:%S.%f000')
    #     str_time = datetime.strftime(d, '%Y-%m-%d %H:%M:%S')
    #     print(str_time)
    #     list_time.append(str_time)

    popup = pd.read_csv(r'C:\Users\user\Desktop\FieldStudy\GREEFA\P13\popup\2019-10-18.csv',
                        names=['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity',
                               'status_popup', 'notes'], sep = ';')
    data['timestamp'] = data.index

    data['timestamp']  = pd.to_datetime(data.index.values, utc=True).tz_convert('Europe/Berlin')
    #df = pd.merge_asof(data, popup, on='timestamp')
    data['timestamp'] = data['timestamp'] + data['timestamp'].apply(lambda x : x.utcoffset())

    popup['timestamp'] = pd.to_datetime(popup['timestamp'], utc = True)
    df_merged = pd.merge(popup, data, on='timestamp', how='outer')


    start = data.index.values[0]
    print(start)
    print(data.index.values)
    print(len(data.index.values))

    start = datetime.strptime(str(start), '%Y-%m-%dT%H:%M:%S.%f000')
    # start = datetime.strftime(start, '%Y-%m-%d %H:%M:%S')
    delta = timedelta(seconds=fs * len(data.index.values))
    end = start + delta
    #t =
    t = pd.to_datetime(data['timestamp'])
    print(t)
    print(len(t))
    # list_time = [datetime.strptime(y, '%Y/%m/%d %H:%M:%S') for y in data.values.tolist()]
    # print(list_time)
    if x_seconds:
        # time_m = np.arange(0, len(data)) / float(sampleRate)
        time_m = t
    else:
        time_m = np.arange(0, len(data)) / (sampleRate * 60.)

    data_min = min(data['EDA'])
    data_max = max(data['EDA'])

    # Plot the data with the Peaks marked

    fig = plt.figure(figsize=(20, 5))

    peak_height = data_max * 1.15
    plt.locator_params(axis='x', nbins=10)
    data['peaks_plot'] = data['peaks'] * peak_height
    plt.plot(time_m, data['peaks_plot'], '#4DBD33')
    # plt.plot(time_m,data['EDA'])
    plt.plot(time_m, data['filtered_eda'])
    # plt.xlim([0, time_m[-1]])
    # plt.xticks(t)

    y_min = min(0, data_min) - (data_max - data_min) * 0.1
    plt.ylim([min(y_min, data_min), peak_height])
    plt.title('EDA with Peaks marked')
    plt.ylabel('$\mu$S')
    ax = plt.gca()

    print(df_merged)
    #df_merged.plot(kind='line', x='activity', y='filtered_eda', ax=ax)

    df_new = df_merged[['timestamp', 'activity']].dropna().set_index('timestamp')
    df_new.index = pd.to_datetime(df_new.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new.iterrows():
        plt.axvline(t, 0,4, c = 'k')
        ax.annotate(a.iloc[0], xy = (t,4), xytext = (t, -1.0), horizontalalignment = 'center', arrowprops = dict(arrowstyle = '-',linestyle = '--', color = 'k'))

    df_new1 = df_merged[['timestamp', 'valence']].dropna().set_index('timestamp')
    df_new1.index = pd.to_datetime(df_new1.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new1.iterrows():
        ax.annotate(str(a.iloc[0]), xy=(t, 4), xytext = (t, -1.5))

    df_new1 = df_merged[['timestamp', 'arousal']].dropna().set_index('timestamp')
    df_new1.index = pd.to_datetime(df_new1.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new1.iterrows():
        ax.annotate(str(a.iloc[0]), xy=(t, 4), xytext = (t, -2.0))


    df_new1 = df_merged[['timestamp', 'dominance']].dropna().set_index('timestamp')
    df_new1.index = pd.to_datetime(df_new1.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new1.iterrows():
        ax.annotate(str(a.iloc[0]), xy=(t, 4), xytext = (t, -2.5))


    fig.subplots_adjust(bottom = 0.3)
    # plt.savefig(r'C:\Users\user\Desktop\FieldStudy\ASML\P18\11-13.png')
    plt.show()



def findPeaks(data, offset=1, start_WT=4, end_WT=4, thres=0.005, sampleRate=SAMPLE_RATE):
    '''
        This function finds the peaks of an EDA signal and returns basic properties.
        Also, peak_end is assumed to be no later than the start of the next peak. (Is this okay??)

        ********* INPUTS **********
        data:        DataFrame with EDA as one of the columns and indexed by a datetimeIndex
        offset:      the number of rising samples and falling samples after a peak needed to be counted as a peak The number of
                    seconds for which the derivative must be positive before a peak and the number of seconds for which
                    the derivative must be negative after a peak.
        start_WT:    maximum number of seconds before the apex of a peak that is the "start" of the peak
        end_WT:      maximum number of seconds after the apex of a peak that is the "rec.t/2" of the peak, 50% of amp
        thres:       the minimum uS change required to register as a peak, defaults as 0 (i.e. all peaks count)
        sampleRate:  number of samples per second, default=8

        ********* OUTPUTS **********
        peaks:               list of binary, 1 if apex of SCR
        peak_start:          list of binary, 1 if start of SCR
        peak_start_times:    list of strings, if this index is the apex of an SCR, it contains datetime of start of peak
        peak_end:            list of binary, 1 if rec.t/2 of SCR
        peak_end_times:      list of strings, if this index is the apex of an SCR, it contains datetime of rec.t/2
        amplitude:           list of floats,  value of EDA at apex - value of EDA at start
        max_deriv:           list of floats, max derivative within 1 second of apex of SCR

    '''
    EDA_deriv = data['filtered_eda'][1:].values - data['filtered_eda'][:-1].values #il succesivo meno il precedente
    peaks = np.zeros(len(EDA_deriv))
    peak_sign = np.sign(EDA_deriv)
    for i in range(int(offset), int(len(EDA_deriv) - offset)):
        if peak_sign[i] == 1 and peak_sign[i + 1] < 1: #zero crossing
            peaks[i] = 1
            for j in range(1, int(offset)):
                if peak_sign[i - j] < 1 or peak_sign[i + j] > -1:
                    # if peak_sign[i-j]==-1 or peak_sign[i+j]==1:
                    peaks[i] = 0
                    break

    # Finding start of peaks
    peak_start = np.zeros(len(EDA_deriv))
    peak_start_times = [''] * len(data)
    max_deriv = np.zeros(len(data))
    rise_time = np.zeros(len(data))

    for i in range(0, len(peaks)):
        if peaks[i] == 1:
            temp_start = max(0, i - sampleRate)
            max_deriv[i] = max(EDA_deriv[temp_start:i])
            start_deriv = .01 * max_deriv[i]

            found = False
            find_start = i
            # has to peak within start_WT seconds
            while found == False and find_start > (i - start_WT * sampleRate):
                if EDA_deriv[find_start] < start_deriv:
                    found = True
                    peak_start[find_start] = 1
                    peak_start_times[i] = data.index[find_start]
                    rise_time[i] = get_seconds_and_microseconds(data.index[i] - pd.to_datetime(peak_start_times[i]))

                find_start = find_start - 1

            # If we didn't find a start
            if found == False:
                peak_start[i - start_WT * sampleRate] = 1
                peak_start_times[i] = data.index[i - start_WT * sampleRate]
                rise_time[i] = start_WT

            # Check if amplitude is too small
            if thres > 0 and (data['EDA'].iloc[i] - data['EDA'][peak_start_times[i]]) < thres:
                peaks[i] = 0
                peak_start[i] = 0
                peak_start_times[i] = ''
                max_deriv[i] = 0
                rise_time[i] = 0

    # Finding the end of the peak, amplitude of peak
    peak_end = np.zeros(len(data))
    peak_end_times = [''] * len(data)
    amplitude = np.zeros(len(data))
    decay_time = np.zeros(len(data))
    half_rise = [''] * len(data)
    SCR_width = np.zeros(len(data))

    for i in range(0, len(peaks)):
        if peaks[i] == 1:
            peak_amp = data['EDA'].iloc[i]
            start_amp = data['EDA'][peak_start_times[i]]
            amplitude[i] = peak_amp - start_amp

            half_amp = amplitude[i] * .5 + start_amp

            found = False
            find_end = i
            # has to decay within end_WT seconds
            while found == False and find_end < (i + end_WT * sampleRate) and find_end < len(peaks):
                if data['EDA'].iloc[find_end] < half_amp:
                    found = True
                    peak_end[find_end] = 1
                    peak_end_times[i] = data.index[find_end]
                    decay_time[i] = get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[i])

                    # Find width
                    find_rise = i
                    found_rise = False
                    while found_rise == False:
                        if data['EDA'].iloc[find_rise] < half_amp:
                            found_rise = True
                            half_rise[i] = data.index[find_rise]
                            SCR_width[i] = get_seconds_and_microseconds(
                                pd.to_datetime(peak_end_times[i]) - data.index[find_rise])
                        find_rise = find_rise - 1

                elif peak_start[find_end] == 1:
                    found = True
                    peak_end[find_end] = 1
                    peak_end_times[i] = data.index[find_end]
                find_end = find_end + 1

            # If we didn't find an end
            if found == False:
                min_index = np.argmin(data['EDA'].iloc[i:(i + end_WT * sampleRate)].tolist())
                peak_end[i + min_index] = 1
                peak_end_times[i] = data.index[i + min_index]

    peaks = np.concatenate((peaks, np.array([0])))
    peak_start = np.concatenate((peak_start, np.array([0])))
    max_deriv = max_deriv * sampleRate  # now in change in amplitude over change in time form (uS/second)

    return peaks, peak_start, peak_start_times, peak_end, peak_end_times, amplitude, max_deriv, rise_time, decay_time, SCR_width, half_rise


def calcPeakFeatures(data, outfile, offset=1, thresh=0.005, start_WT=4, end_WT=4):
    returnedPeakData = findPeaks(data, offset * SAMPLE_RATE, start_WT, end_WT, thresh, SAMPLE_RATE)
    data['peaks'] = returnedPeakData[0]
    data['peak_start'] = returnedPeakData[1]
    data['peak_end'] = returnedPeakData[3]

    data['peak_start_times'] = returnedPeakData[2]
    data['peak_end_times'] = returnedPeakData[4]
    data['half_rise'] = returnedPeakData[10]
    # Note: If an SCR doesn't decrease to 50% of amplitude, then the peak_end = min(the next peak's start, 15 seconds after peak)
    data['amp'] = returnedPeakData[5]
    data['max_deriv'] = returnedPeakData[6]
    data['rise_time'] = returnedPeakData[7]
    data['decay_time'] = returnedPeakData[8]
    data['SCR_width'] = returnedPeakData[9]

    # To keep all filtered data remove this line
    featureData = data[data.peaks == 1][['EDA', 'rise_time', 'max_deriv', 'amp', 'decay_time', 'SCR_width']]

    # Replace 0s with NaN, this is where the 50% of the peak was not found, too close to the next peak
    featureData[['SCR_width', 'decay_time']] = featureData[['SCR_width', 'decay_time']].replace(0, np.nan)
    featureData['AUC'] = featureData['amp'] * featureData['SCR_width']

    featureData.to_csv(outfile)

    return data

fullOutputPath = r'C:\Users\user\Desktop\FieldStudy\ASML\P18\week1\10_31_2019, 12_00_38\find_peak.csv'
filepath = r'C:\Users\user\Desktop\FieldStudy\GREEFA\P13\week2\10_18_2019, 08_16_50'
filepath_confirm = os.path.join(filepath, "EDA.csv")
data = loadData_E4(filepath)
#data = detect_fast_edges(data)
returnedPeakData = findPeaks(data)
peakData = calcPeakFeatures(data, fullOutputPath)
plotPeaks(peakData)



