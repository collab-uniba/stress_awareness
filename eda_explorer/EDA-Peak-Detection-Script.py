import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pprint
from datetime import datetime, timedelta
from dateutil import tz
from load_files import getInputLoadFile, get_user_input, getOutputPath
import sys
import matplotlib.patches as mpatches

SAMPLE_RATE = 8


def findPeaks(data, offset, start_WT, end_WT, thres=0, sampleRate=SAMPLE_RATE):
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
    print()
    EDA_deriv = data['filtered_eda'][1:].values - data['filtered_eda'][:-1].values #il succesivo meno il precedente
    print(data.head())
    print(EDA_deriv)
    peaks = np.zeros(len(EDA_deriv))
    peak_sign = np.sign(EDA_deriv)
    for i in range(int(offset), int(len(EDA_deriv) - offset)):
        if peak_sign[i] == 1 and peak_sign[i + 1] < 1:
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
            if thres > 0 and (data['filtered_eda'].iloc[i] - data['filtered_eda'][peak_start_times[i]]) < thres:
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
            peak_amp = data['filtered_eda'].iloc[i]
            start_amp = data['filtered_eda'][peak_start_times[i]]
            amplitude[i] = peak_amp - start_amp

            half_amp = amplitude[i] * .5 + start_amp

            found = False
            find_end = i
            # has to decay within end_WT seconds
            while found == False and find_end < (i + end_WT * sampleRate) and find_end < len(peaks):
                if data['filtered_eda'].iloc[find_end] < half_amp:
                    found = True
                    peak_end[find_end] = 1
                    peak_end_times[i] = data.index[find_end]

                    #start = datetime.strptime(str(start), '%Y-%m-%dT%H:%M:%S.%f000')
                    decay_time[i] = get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[i])

                    # Find width
                    find_rise = i
                    found_rise = False
                    while found_rise == False:
                        if data['filtered_eda'].iloc[find_rise] < half_amp:
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
                min_index = np.argmin(data['filtered_eda'].iloc[i:(i + end_WT * sampleRate)].tolist())
                peak_end[i + min_index] = 1
                peak_end_times[i] = data.index[i + min_index]

    peaks = np.concatenate((peaks, np.array([0])))
    peak_start = np.concatenate((peak_start, np.array([0])))
    max_deriv = max_deriv * sampleRate  # now in change in amplitude over change in time form (uS/second)

    return peaks, peak_start, peak_start_times, peak_end, peak_end_times, amplitude, max_deriv, rise_time, decay_time, SCR_width, half_rise


def get_seconds_and_microseconds(pandas_time):
    return pandas_time.seconds + pandas_time.microseconds * 1e-6


def calcPeakFeatures(data, outfile, offset, thresh, start_WT, end_WT):
    returnedPeakData = findPeaks(data, offset * 8, start_WT, end_WT, thresh, 8)
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
    featureData = data[data.peaks == 1][['filtered_eda', 'rise_time', 'max_deriv', 'amp', 'decay_time', 'SCR_width']]

    # Replace 0s with NaN, this is where the 50% of the peak was not found, too close to the next peak
    featureData[['SCR_width', 'decay_time']] = featureData[['SCR_width', 'decay_time']].replace(0, np.nan)
    featureData['AUC'] = featureData['amp'] * featureData['SCR_width']

    featureData.to_csv(outfile)

    return data

from dateutil import parser
# draws a graph of the data with the peaks marked on it
# assumes that 'data' dataframe already contains the 'peaks' column
def plotPeaks(data, x_seconds, sampleRate=SAMPLE_RATE):
    list_time = []
    fs = 8
    # for y in data.index.values:
    #     print(y)
    #     d = datetime.strptime(str(y), '%Y-%m-%dT%H:%M:%S.%f000')
    #     str_time = datetime.strftime(d, '%Y-%m-%d %H:%M:%S')
    #     print(str_time)
    #     list_time.append(str_time)
    names = ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity',
             'status_popup', 'notes']
    path_popup = sys.argv[3]
    hours_init = 10
    hours_end = 18
    popup = pd.read_csv(path_popup, names=names,
                      sep=';')
    print(popup)
    data['timestamp'] = pd.to_datetime(data.index.values, utc=True).tz_convert('Europe/Berlin')
    # df = pd.merge_asof(data, popup, on='timestamp')
    data['timestamp'] = data['timestamp'] + data['timestamp'].apply(lambda x: x.utcoffset())
    popup['timestamp'] = pd.to_datetime(popup['timestamp'], utc=True)

    data['hours'] = data['timestamp'].dt.hour
    data = data[data['hours'].between(hours_init, hours_end)]

    popup['hours'] = popup['timestamp'].dt.hour
    popup = popup[popup['hours'].between(hours_init - 2, hours_end - 2)]

    df_merged = pd.merge(popup, data, on='timestamp', how='outer')
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'].values, utc=True)
    #df_merged['hours'] = df_merged['timestamp'].dt.hour
    #df_merged = df_merged[df_merged['hours'].between(hours_init - 2, hours_end - 2)]
    df_merged = df_merged.sort_values(by='timestamp')
    df_merged.to_csv('merged.csv')
    #df_merged = df_merged[df_merged['filtered_eda'].notna()]

    t = pd.to_datetime(data['timestamp'])
    # list_time = [datetime.strptime(y, '%Y/%m/%d %H:%M:%S') for y in data.values.tolist()]
    # print(list_time)
    if x_seconds:
        # time_m = np.arange(0, len(data)) / float(sampleRate)
        time_m = t


    data_min = min(data['filtered_eda'])
    data_max = max(data['filtered_eda'])


    # Plot the data with the Peaks marked

    fig = plt.figure(figsize=(20, 5))

    peak_height = data_max * 1.15
    plt.locator_params(axis='x', nbins=10)
    df_merged['peaks_plot'] = df_merged['peaks'] * peak_height
    #plt.plot(time_m, df_merged['peaks_plot'], 'orange')

    # plt.plot(time_m,data['EDA'])
    plt.plot(time_m, data['filtered_eda'])


    y_min = min(0, data_min) - (data_max - data_min) * 0.1
    plt.ylim([min(y_min, data_min),peak_height ])
    plt.title('EDA with Peaks marked as vertical lines')
    plt.ylabel('$\mu$S')
    ax = plt.gca()


    red_patch = mpatches.Patch(color='#FF0000', label='High')
    orange_patch = mpatches.Patch(color='#FF8C00', label='Average')
    green_patch = mpatches.Patch(color='#4DBD33', label='Low')
    plt.legend(handles=[red_patch, orange_patch, green_patch], loc = 'upper left')

    df_peak = df_merged[['timestamp', 'peaks_plot', 'arousal']].set_index('timestamp')
    df_peak = df_peak.fillna(method= 'backfill')
    df_peak = df_peak.fillna(method='ffill')
    df_peak = df_peak.loc[~((df_peak['peaks_plot'] == 0))]

    print('df_peak', df_peak)
    for t, a in df_peak.iterrows():
        if a['arousal'] == 1 or a['arousal'] == 2:
            color = '#4DBD33'
        elif a['arousal'] == 3:
            color = '#FF8C00'
        else:
            color = '#FF0000'

        plt.axvline(t, color=color, linestyle='-', alpha = 0.5)



    # df_merged.plot(kind='line', x='activity', y='filtered_eda', ax=ax)
    y = calc_y(peak_height, 25)
    df_new = df_merged[['timestamp', 'activity']].dropna().set_index('timestamp')
    df_new.index = pd.to_datetime(df_new.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new.iterrows():
        plt.axvline(t, 0, 4, c='#808080', linestyle='--', linewidth=0.5)
        ax.annotate(a.iloc[0], xy=(t, peak_height), xytext=(t, y[0]), horizontalalignment='center',
                    arrowprops=dict(arrowstyle='-', linestyle='--', color='#808080', linewidth=0.5))

    df_new1 = df_merged[['timestamp', 'valence']].dropna().set_index('timestamp')
    df_new1.index = pd.to_datetime(df_new1.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new1.iterrows():
        if a.iloc[0] == 1 or a.iloc[0] == 2:
            color = '#4DBD33'
        elif a.iloc[0] == 3:
            color = '#FF8C00'
        else:
            color = '#FF0000'
        ax.annotate(str(a.iloc[0]), xy=(t, peak_height), xytext=(t, y[1]), color = color)

    df_new1 = df_merged[['timestamp', 'arousal']].dropna().set_index('timestamp')
    df_new1.index = pd.to_datetime(df_new1.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new1.iterrows():
        if a.iloc[0] == 1 or a.iloc[0] == 2:
            color = '#4DBD33'
        elif a.iloc[0] ==  3:
            color = '#FF8C00'
        else:
            color = '#FF0000'
        ax.annotate(str(a.iloc[0]), xy=(t, peak_height), xytext=(t, y[2]), color = color)

    df_new1 = df_merged[['timestamp', 'dominance']].dropna().set_index('timestamp')
    df_new1.index = pd.to_datetime(df_new1.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new1.iterrows():
        if a.iloc[0] ==1 or a.iloc[0] == 2:
            color = '#4DBD33'
        elif a.iloc[0] == 3:
            color = '#FF8C00'
        else:
            color = '#FF0000'
        ax.annotate(str(a.iloc[0]), xy=(t, peak_height), xytext=(t,y[3]), color=color)

    df_new1 = df_merged[['timestamp', 'productivity']].dropna().set_index('timestamp')
    df_new1.index = pd.to_datetime(df_new1.index.values, utc=True).tz_convert('Europe/Berlin')
    for t, a in df_new1.iterrows():
        ax.annotate(str(a.iloc[0]), xy=(t, peak_height), xytext=(t, y[4]))

    trans = ax.get_yaxis_transform()  # x in data untis, y in axes fraction
    y_labels = -0.08
    ax.annotate('Activity', xy=(y_labels, y[0]), color = '#1f77b4',xycoords=trans)
    ax.annotate('Valence', xy=(y_labels, y[1]),color = '#1f77b4', xycoords=trans)
    ax.annotate('Arousal', xy=(y_labels, y[2]), color = '#1f77b4', xycoords=trans)
    ax.annotate('Dominance', xy=(y_labels, y[3]), color = '#1f77b4', xycoords=trans)
    ax.annotate('Productivity', xy=(y_labels, y[4]),color = '#1f77b4', xycoords=trans)


    fig.subplots_adjust(bottom=0.4, left=0.1)
    # plt.savefig(r'C:\Users\user\Desktop\FieldStudy\ASML\P18\11-13.png')
    plt.show()


def chooseValueOrDefault(str_input, default):
    if str_input == "":
        return default
    else:
        return float(str_input)

def calc_y(a,c ):
    b = []
    for _ in range(0, 5):
        y = (a * c / 100)
        b.append(-y)
        c += 10
    return b

if __name__ == "__main__":

    path_to_E4 = sys.argv[1]
    data = pd.read_csv(path_to_E4)
    data.index = data['timestamp']
    data.index = pd.to_datetime(data.index.values)
    #data['timestamp'] = data.index

    #print("index: ", data.index)
    filepath_confirm = path_to_E4

    #fullOutputPath = getOutputPath()

    #print("")
    #print("Please choose settings for the peak detection algorithm. For default values press return")
    # thresh_str = get_user_input('\tMinimum peak amplitude (default = .02):')
    # thresh = chooseValueOrDefault(thresh_str, .02)
    # offset_str = get_user_input('\tOffset (default = 1): ')
    # offset = chooseValueOrDefault(offset_str, 1)
    # start_WT_str = get_user_input('\tMax rise time (s) (default = 4): ')
    # start_WT = chooseValueOrDefault(start_WT_str, 4)
    # end_WT_str = get_user_input('\tMax decay time (s) (default = 4): ')
    # end_WT = chooseValueOrDefault(end_WT_str, 4)

    #input_path = "C:/Users/user/Desktop/FieldStudy/ASML/P18/week2/11_04_2019,10_28_58/"
    artifact_output_path = "C:/Users/user/Desktop/FieldStudy/ASML/P18/week2/11_04_2019,10_28_58/"
    fullOutputPath = sys.argv[2] #artifact_output_path + 'result_peak.csv'



    thresh = .1
    offset = 1
    start_WT = 4
    end_WT = 4

    settings_dict = {'threshold': thresh,
                     'offset': offset,
                     'rise time': start_WT,
                     'decay time': end_WT}

    print("")
    print("Okay, finding peaks in file " + filepath_confirm + " using the following parameters")
    pprint.pprint(settings_dict)
    peakData = calcPeakFeatures(data, fullOutputPath, offset, thresh, start_WT, end_WT)
    print("Features computed and saved to " + fullOutputPath)
    plotPeaks(peakData, True)

