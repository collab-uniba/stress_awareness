import os
import scipy.signal as scisig
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from bokeh.io import output_file

SAMPLE_RATE = 8
import zipfile
from base64 import b64decode
from io import BytesIO
import panel as pn
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Span, Label, HoverTool, Div, FileInput
from bokeh.layouts import column

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


def calc_y(a,c ):
    b = []
    for _ in range(0, 6):
        y = (a * c / 100)
        b.append(-y)
        c += 10
    return b
def percentage(part, whole):
  return float(whole) * float(part)/100



def plotPeaks(data, x_seconds=True, sampleRate=8):

 
    list_time = []
    fs = 8
    print(data['filtered_eda'].empty)
    names = ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity',
             'status_popup', 'notes', 'filtered_eda']
    popup = pd.read_csv(r'C:/Users/Daniela/Desktop/EmoVizPhy/stress_awareness/stress_awareness/LabStudy/S1/popup/popup/2022-06-02.csv', sep = ';')
    
    print(popup)
  
    data['timestamp'] = pd.to_datetime(data.index.values, utc=True).tz_convert('Europe/Berlin')
    print(data['timestamp'])
    data['timestamp'] = data['timestamp'] + data['timestamp'].apply(lambda x: x.utcoffset())
    popup['timestamp'] = pd.to_datetime(popup['timestamp'], utc=True)
    data['hours'] = data['timestamp'].dt.hour
    popup['hours'] = popup['timestamp'].dt.hour
    df_merged = pd.merge(popup, data, on='timestamp', how='outer')
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'].values, utc=True)
    df_merged = df_merged.sort_values(by='timestamp')
    df_merged.to_csv('merged.csv')
    #df_merged = df_merged[df_merged['filtered_eda'].notna()]
    t = pd.to_datetime(data['timestamp'])
    # list_time = [datetime.strptime(y, '%Y/%m/%d %H:%M:%S') for y in data.values.tolist()]
    # print(list_time)
    if x_seconds:
        # time_m = np.arange(0, len(data)) / float(sampleRate)
        time_m = t

    print(data['filtered_eda'].empty)
    data_min = min(data['filtered_eda'])

    data_max = max(data['filtered_eda'])

    pn.extension()


    df_popup = df_merged[['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity', 'notes', 'filtered_eda']]
    # drop the row if the column activity is nan
    df_popup = df_popup[df_popup['activity'].notna()]
    print(df_popup)
    # plot df_popup in another fig
    datasrc = ColumnDataSource(df_popup)

    # Define the figure and its parameters
    fig = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
                    title='EDA with Peaks marked as vertical lines', x_axis_label='Time', y_axis_label='$\mu$S',sizing_mode = 'stretch_both')

    # Define the data source
    data_src = ColumnDataSource(df_merged)

  
    line_plot = fig.line(x='timestamp', y='filtered_eda', source=data_src)
    circle_plot = fig.circle(name = 'report', x='timestamp', y='filtered_eda', source=datasrc, fill_color="white", size= 8)

    line_hover = HoverTool(renderers=[line_plot], tooltips=[("EDA", "@filtered_eda"), ("Timestamp", "@timestamp{%F}")], formatters={'@timestamp': 'datetime'})
    circle_hover = HoverTool(renderers=[circle_plot], tooltips=[("Activity", "@activity"), ("Valence", "@valence"), ("Arousal", "@arousal"), ("Dominance", "@dominance"), ("Productivity", "@productivity"), ("Notes", "@notes"), ("Timestamp", "@timestamp{%F}")], formatters={'@timestamp': 'datetime'})
    fig.add_tools(line_hover, circle_hover)


    #Add the peak markers to the figure
    peak_height = data['filtered_eda'].max() * 1.15
    df_merged['peaks_plot'] = df_merged['peaks'] * peak_height
    df_peak = df_merged[['timestamp', 'peaks_plot', 'arousal']].set_index('timestamp')
    df_peak = df_peak.fillna(method='backfill').fillna(method='ffill').loc[~((df_peak['peaks_plot'] == 0))]
    for t, a in df_peak.iterrows():
        if a['arousal'] == 1 or a['arousal'] == 2:
            color = '#4DBD33'
        elif a['arousal'] == 3:
            color = '#FF8C00'
        else:
            color = '#FF0000'
        fig.add_layout(Span(location=t, dimension='height', line_color=color, line_alpha=0.5, line_width=1))


    #add below the first plot aonther plot with another signal
    fig2 = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
    title='EDA with Peaks marked as vertical lines', x_axis_label='Time', y_axis_label='$\mu$S')

    fig2.line(x='timestamp', y='filtered_eda', source=data_src)

    fig3 = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
    title='EDA with Peaks marked as vertical lines', x_axis_label='Time', y_axis_label='$\mu$S')

    fig3.line(x='timestamp', y='filtered_eda', source=data_src)

    fig4 = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
    title='EDA with Peaks marked as vertical lines', x_axis_label='Time', y_axis_label='$\mu$S')

    fig4.line(x='timestamp', y='filtered_eda', source=data_src)

    fig4.add_layout(Label(text="Bottom Centered Title"), "below")





    show(column(fig, fig2, fig3, fig4))



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

fullOutputPath = r'C:/Users/Daniela/Desktop/EmoVizPhy/stress_awareness/stress_awareness/LabStudy/S1/Data/2giugno2022_150205_A02886/find_peak.csv'
filepath = r'C:/Users/Daniela/Desktop/EmoVizPhy/stress_awareness/stress_awareness/LabStudy/S1/Data/2giugno2022_150205_A02886'
filepath_confirm = os.path.join(filepath, "EDA.csv")
data = loadData_E4(filepath)
#data = detect_fast_edges(data)
returnedPeakData = findPeaks(data)
peakData = calcPeakFeatures(data, fullOutputPath)
plotPeaks(peakData)



