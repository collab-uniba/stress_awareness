import pprint
import sys
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Span, Label, HoverTool
from bokeh.layouts import column
SAMPLE_RATE = 8
import panel as pn

def findPeaks(data, offset, start_WT, end_WT, thres=0, sampleRate=SAMPLE_RATE):
    '''
        This function finds the peaks of an EDA signal and returns basic properties.
        Also, peak_end is assumed to be no later than the start ofeh lo  the next peak. (Is this okay??)

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
            print(data['filtered_eda'].empty)
            amplitude[i] = peak_amp - start_amp

            half_amp = amplitude[i] * .5 + start_amp

            found = False
            find_end = i
            # has to decay within end_WT seconds
            while found == False and find_end < (i + end_WT * sampleRate) and find_end < len(peaks):
                if data['filtered_eda'].iloc[find_end] < half_amp:
                    print(data['filtered_eda'].empty)
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
                            print(data['filtered_eda'].empty)
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
            print(data['filtered_eda'])
            if found == False:
                min_index = np.argmin(data['filtered_eda'].iloc[i:(i + end_WT * sampleRate)].tolist())
                print(data['filtered_eda'].empty)
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


def plotPeaks(data, x_seconds, sampleRate=SAMPLE_RATE):
    list_time = []
    fs = 8
    print(data['filtered_eda'].empty)
    names = ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity',
             'status_popup', 'notes', 'filtered_eda']
    popup = pd.read_csv(
        r'..//LabStudy/S1/popup/popup/2022-06-02.csv',
        sep=';')

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
    # df_merged = df_merged[df_merged['filtered_eda'].notna()]
    t = pd.to_datetime(data['timestamp'])
    if x_seconds:
        # time_m = np.arange(0, len(data)) / float(sampleRate)
        time_m = t

    print(data['filtered_eda'].empty)
    data_min = min(data['filtered_eda'])

    data_max = max(data['filtered_eda'])

    pn.extension()

    df_popup = df_merged[
        ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity', 'notes', 'filtered_eda']]
    # drop the row if the column activity is nan
    df_popup = df_popup[df_popup['activity'].notna()]
    print("df_popup")
    print(df_popup)
    # plot df_popup in another fig
    datasrc = ColumnDataSource(df_popup)

    # Define the figure and its parameters
    fig = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
                 title='EDA with Peaks marked as vertical lines', x_axis_label='Time', y_axis_label='$\mu$S',
                 sizing_mode='stretch_both')

    # Define the data source
    data_src = ColumnDataSource(df_merged)

    line_plot = fig.line(x='timestamp', y='filtered_eda', source=data_src)
    circle_plot = fig.circle(name='report', x='timestamp', y='filtered_eda', source=datasrc, fill_color="white", size=8)

    line_hover = HoverTool(renderers=[line_plot], tooltips=[("EDA", "@filtered_eda"), ("Timestamp", "@timestamp{%F}")],
                           formatters={'@timestamp': 'datetime'})
    circle_hover = HoverTool(renderers=[circle_plot],
                             tooltips=[("Activity", "@activity"), ("Valence", "@valence"), ("Arousal", "@arousal"),
                                       ("Dominance", "@dominance"), ("Productivity", "@productivity"),
                                       ("Notes", "@notes"), ("Timestamp", "@timestamp{%F}")],
                             formatters={'@timestamp': 'datetime'})
    fig.add_tools(line_hover, circle_hover)

    # Add the peak markers to the figure
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

    # add below the first plot aonther plot with another signal
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


def chooseValueOrDefault(str_input, default):
    if str_input == "":
        return default
    else:
        return float(str_input)


def calc_y(a,c ):
    b = []
    for _ in range(0, 6):
        y = (a * c / 100)
        b.append(-y)
        c += 10
    return b


if __name__ == "__main__":

    path_to_E4 = sys.argv[1]
    data = pd.read_csv(path_to_E4)
    data.index = data['timestamp']
    data.index = pd.to_datetime(data.index.values)
    filepath_confirm = path_to_E4

    fullOutputPath = sys.argv[2]


    thresh = .2
    offset = 2
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

