import datetime
import numpy as np
import glob
import importlib
import sys
import constant
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure


sys.path.insert(1, './signalPreprocess')
eda_artifact = importlib.import_module("EDA-Artifact-Detection-Script")
eda_peak = importlib.import_module("EDA-Peak-Detection-Script")
accelerometer = importlib.import_module("ACC_HR_Filtering")

def classify_artifacts(eda, acc,temp, fullOutputPath, ouput_path):
    labels, data = eda_artifact.classify(constant.classifierList, eda, acc, temp)

    featureLabels = pd.DataFrame(labels, index=pd.date_range(start=data.index[0], periods=len(labels), freq='5s'),
                      columns=constant.classifierList)
    featureLabels.reset_index(inplace=True)
    featureLabels.rename(columns={'index': 'StartTime'}, inplace=True)
    featureLabels['EndTime'] = featureLabels['StartTime'] + datetime.timedelta(seconds=5)
    featureLabels.index.name = 'EpochNum'
    cols = ['StartTime', 'EndTime']
    cols.extend(constant.classifierList)
    featureLabels = featureLabels[cols]
    featureLabels.rename(columns={'Binary': 'BinaryLabels', 'Multiclass': 'MulticlassLabels'},
                         inplace=True)
    featureLabels.to_csv(fullOutputPath)
    data.to_csv(ouput_path)


def detect_peak(ouput_path, artifact_path, thresh, offset, start_WT, end_WT):
    signal_df = pd.read_csv(ouput_path,  names = ['timestamp', 'EDA', 'filtered_eda', 'AccelX', 'AccelY', 'AccelZ', 'Temp'])
    artifact_df = pd.read_csv(artifact_path)
    signal_df['timestamp'] = signal_df['timestamp'].astype('datetime64[ns]')
    artifact_df['StartTime'] = artifact_df['StartTime'].astype('datetime64[ns]')
    eda_clean = pd.merge(signal_df, artifact_df, how = 'outer', left_on='timestamp', right_on='StartTime')
    eda_clean = eda_clean.fillna(method = 'ffill')
    x = eda_clean['filtered_eda'].values
    dx = eda_clean['BinaryLabels']
    filt = [np.nan if t == -1.0 else y for y,t in zip(x,dx)]
    eda_clean['filtered_eda'] = filt
    eda_clean['filtered_eda'] = eda_clean['filtered_eda'].ffill()
    eda_clean = eda_clean[~eda_clean['filtered_eda'].isin(['filtered_eda'])]
    final_df = eda_clean[['timestamp', 'filtered_eda']]
    final_df.to_csv(r"./temp" + '/filtered_eda.csv', index = False)
    path_to_E4 = r"./temp" + "/filtered_eda.csv"
    data = pd.read_csv(path_to_E4)
    data.index = data['timestamp']
    data.index = pd.to_datetime(data.index.values)
    fullOutputPath = r"./temp" + "/result_peak.csv"

    return eda_peak.calcPeakFeatures(
        data, fullOutputPath, float(offset.value), float(thresh.value), int(start_WT.value), int(end_WT.value)
    )

def get_datetime_filename(column):
    human_timestamp = []
    for value in column:
        human_date = datetime.datetime.fromtimestamp(int(value))
        human_timestamp.append(human_date)
    return human_timestamp


def uniform_csv(filename):
    with open(filename, 'r') as file:
        filedata = file.read()
        filedata = filedata.replace(';', ',')
        filedata = filedata.replace('Timestamp,Activity,Valence,Arousal,Dominance,Progress,Status,Notes', '')

    with open(filename, 'w') as file:
        file.write(filedata)

def calculate_date_time(timestamp_0, hz, num_rows):
    format = "%d/%m/%Y, %H:%M:%S"
    date_time_0 = datetime.datetime.fromtimestamp(timestamp_0)
    #Change datatime format
    date_time_0_str = date_time_0.strftime(format)
    date_time_0 = datetime.datetime.strptime(date_time_0_str, format)
    data_times = [date_time_0]
    off_set = 1 / hz
    for i in range(1, num_rows):
        data_time_temp = data_times[i-1] + datetime.timedelta(seconds = off_set)
        data_times.append(data_time_temp)
    date = str(data_times[0].date())
    times = [t.strftime("%H:%M:%S") for t in data_times]
    return date, times


def popup_process():
    path = r"./temp"
    all_files = glob.glob(f"{path}/*data*.csv")
    df_list = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None,
                         names=['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity',
                                'status_popup', 'notes'])
        df_list.append(df)

    frame = pd.concat(df_list, axis=0, ignore_index=True)
    frame = frame.drop_duplicates()
    first_column = frame['timestamp']
    frame['timestamp'] = get_datetime_filename(first_column)

    frame['arousal'] = np.select(
        [((frame['valence'] == 4.0) | (frame['valence'] == 5.0)) & (
                    (frame['arousal'] == 1.0) | (frame['arousal'] == 2.0)),
         ((frame['valence'] == 4.0) | (frame['valence'] == 5.0)) & (
                     (frame['arousal'] == 4.0) | (frame['arousal'] == 5.0)),
         ((frame['valence'] == 1.0) | (frame['valence'] == 2.0)) & (
                     (frame['arousal'] == 4.0) | (frame['arousal'] == 5.0)),
         ((frame['valence'] == 1.0) | (frame['valence'] == 2.0)) & (
                     (frame['arousal'] == 1.0) | (frame['arousal'] == 2.0)),
         (frame['arousal'] == 3.0)],
        ['Low 🧘‍♀', 'High 🤩', 'High 😤', 'Low 😔', 'Medium 😐'], default='Unknown'
    )
    convert_to_discrete(frame, 'valence')
    convert_to_discrete(frame, 'dominance')

    return frame


def convert_to_discrete(frame, column):
    replacements = {
        'valence': {1.0: 'Low 😔', 2.0: 'Low 😔', 3.0: 'Medium 😐', 4.0: 'High 😄', 5.0: 'High 😄'},
        'dominance': {1.0: 'Low 😔🥱', 2.0: 'Low 😔🥱', 3.0: 'Medium 😐',  4.0: 'High 👨‍🎓', 5.0: 'High 👨‍🎓'},
    }
    frame[column] = frame[column].replace(replacements[column])

def process_data_popup(data, popup):
    data['timestamp'] = pd.to_datetime(data.index.values, utc=True).tz_convert('Europe/Berlin')
    data['timestamp'] = data['timestamp'] + data['timestamp'].apply(lambda x: x.utcoffset())
    
    #Rimozione dei popup relativi agli altri giorni
    data.reset_index(inplace=True, drop=True)
    date = data.loc[0, 'timestamp']
    popup = extract_popup_date(popup, date)


    popup['timestamp'] = pd.to_datetime(popup['timestamp'], utc=True)
    data['hours'] = data['timestamp'].dt.hour
    popup['hours'] = popup['timestamp'].dt.hour
    df_merged = pd.merge(popup, data, on='timestamp', how='outer')
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'].values, utc=True)
    df_merged = df_merged.sort_values(by='timestamp')
    return df_merged

def process_acc():
    acc, timestamp_0 = accelerometer.load_acc()
    acc_filter = accelerometer.empatica_filter(acc)
    date, time = calculate_date_time(timestamp_0,1,len(acc_filter))
    # create a df with the filtered acc data and date and time
    df_acc = pd.DataFrame(acc_filter, columns=['acc_filter'])
    df_acc['time'] = time
    df_acc['date'] = date
    df_acc['timestamp'] = pd.to_datetime(df_acc['date'] + ' ' + df_acc['time'])
    return df_acc

def process_hr():
    hr, timestamp_0_hr = accelerometer.load_hr()
    date, time = calculate_date_time(timestamp_0_hr, 1, len(hr))
    df_hr = pd.DataFrame(hr, columns=['hr'])
    df_hr['time'] = time
    df_hr['date'] = date
    df_hr['timestamp'] = pd.to_datetime(df_hr['date'] + ' ' + df_hr['time'])

    return df_hr


def create_fig_line(df_sign, x, y, title, y_axis_label, sign, df_popup = None):
    fig_sign = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
                    title=title, x_axis_label='Time', y_axis_label=y_axis_label,
                    sizing_mode='stretch_both')
    data_src_sign = ColumnDataSource(df_sign)
    line_plot_sign = fig_sign.line(x=x, y=y, source=data_src_sign)
    

    line_hover = HoverTool(renderers=[line_plot_sign],
                            tooltips=[(sign, "@"+y), ("Time", "@timestamp{%H:%M:%S}")],
                            formatters={'@timestamp': 'datetime'})
    fig_sign.add_tools(line_hover)

    
    #Mean
    mean = df_sign.loc[:, y].mean()
    fig_sign.add_layout(Span(location=mean, dimension='width', line_color="red", line_alpha=0.5, line_width=1, line_dash='dashed'))
    

    
    
    
    if sign == 'EDA':
        datasrc = ColumnDataSource(df_popup)
        circle_plot = fig_sign.circle(name='report', x=x, y=y, source=datasrc, fill_color="yellow",
                               size=9)
        circle_hover = HoverTool(renderers=[circle_plot],
                                tooltips=[("Activity", "@activity"), ("Valence", "@valence"), ("Arousal", "@arousal"),
                                        ("Dominance", "@dominance"), ("Productivity", "@productivity"),
                                        ("Notes", "@notes"), ("Timestamp", "@timestamp{%H:%M:%S}"), (sign, "@"+y)],
                                formatters={'@timestamp': 'datetime'})
        fig_sign.add_tools(circle_hover)
    
    
    return fig_sign

def extract_popup_date(popup, timestamp):
    #Rimuove i popup relativi agli altri giorni
    date = timestamp.date()
    popup.reset_index(inplace=True, drop=True)
    for i in range(popup.shape[0]):
        date_temp = popup.loc[i, 'timestamp'].date()

        if date_temp != date:
            popup.drop(i, inplace=True)
            
    popup.reset_index(inplace=True, drop=True)
    return popup