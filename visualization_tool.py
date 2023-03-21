import csv
import glob
import os
import datetime
import numpy as np
import pandas as pd
import importlib
import sys
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Span, HoverTool
from bokeh.layouts import column
import io
import zipfile
sys.path.insert(1, './eda_explorer')
eda_artifact = importlib.import_module("EDA-Artifact-Detection-Script")
eda_peak = importlib.import_module("EDA-Peak-Detection-Script")
SAMPLE_RATE = 8
import panel as pn
from panel.widgets import FileInput

'''
 Please note that this script use scripts released by Taylor et al. that you can find here: https://github.com/MITMediaLabAffectiveComputing/eda-explorer
 
  Taylor, Sara et al. “Automatic identification of artifacts in electrodermal activity data.” 
  Annual International Conference of the IEEE Engineering in Medicine and Biology Society. 
  IEEE Engineering in Medicine and Biology Society. 
  Annual International Conference vol. 2015 (2015): 1934-7. 
  doi:10.1109/EMBC.2015.7318762

'''
numClassifiers = 1
classifierList = ['Binary']
thresh = .2
offset = 2
start_WT = 4
end_WT = 4
artifact_output_path = r"./temp"
file_upload = FileInput()
bokeh_pane = pn.pane.Bokeh()

def classify_artifacts(classifierList, eda, acc,temp, fullOutputPath, ouput_path):
    labels, data = eda_artifact.classify(classifierList, eda, acc, temp)

    featureLabels = pd.DataFrame(labels, index=pd.date_range(start=data.index[0], periods=len(labels), freq='5s'),
                                 columns=classifierList)
    featureLabels.reset_index(inplace=True)
    featureLabels.rename(columns={'index': 'StartTime'}, inplace=True)
    featureLabels['EndTime'] = featureLabels['StartTime'] + datetime.timedelta(seconds=5)
    featureLabels.index.name = 'EpochNum'
    cols = ['StartTime', 'EndTime']
    cols.extend(classifierList)
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
    final_df.to_csv(r"./eda_explorer/temp" + '/filtered_eda.csv', index = False)
    path_to_E4 = r"./eda_explorer/temp" + "/filtered_eda.csv"
    data = pd.read_csv(path_to_E4)
    data.index = data['timestamp']
    data.index = pd.to_datetime(data.index.values)
    fullOutputPath = r"./eda_explorer/temp" + "/result_peak.csv"

    return eda_peak.calcPeakFeatures(
        data, fullOutputPath, offset, thresh, start_WT, end_WT
    )


def get_datetime_filename(column):
    human_timestamp = []
    for value in column:
        human_date = datetime.datetime.fromtimestamp(int(value))
        human_timestamp.append(human_date)
    return human_timestamp

def uniform_csv(filename):
    # Read in the file
    with open(filename, 'r') as file:
      filedata = file.read()
    # Replace the target string
    filedata = filedata.replace(';', ',')
    filedata = filedata.replace('Timestamp,Activity,Valence,Arousal,Dominance,Progress,Status,Notes', '')
    #print(filedata)
    # Write the file out again
    with open(filename, 'w') as file:
      file.write(filedata)


def popup_process():
    #read all file in a specific folder
    path = r"./temp"
    all_files = glob.glob(f"{path}/*data*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, names=['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity', 'status_popup','notes'])

        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    # delete duplicate rows
    frame = frame.drop_duplicates()
    first_column = frame['timestamp']
    frame['timestamp'] = get_datetime_filename(first_column)
    print(frame['arousal'])
    frame.loc[(frame['valence'] == 4.0) | (frame['valence'] == 5.0) & (frame['arousal'] == 1.0) | (frame['arousal'] == 2.0), 'arousal'] = 'Low 😔'
    frame.loc[(frame['valence'] == 4.0) | (frame['valence'] == 5.0) & (frame['arousal'] == 4.0) | (frame['arousal'] == 5.0), 'arousal'] = 'High 🤩'
    frame.loc[(frame['valence'] == 1.0) | (frame['valence'] == 2.0) & (frame['arousal'] == 4.0) | (frame['arousal'] == 5.0), 'arousal'] = 'High 😤'
    frame.loc[(frame['valence'] == 1.0) | (frame['valence'] == 2.0) & (frame['arousal'] == 1.0) | (frame['arousal'] == 2.0), 'arousal'] = 'Low 🧘‍♀'
    frame.loc[(frame['arousal'] == 3.0), 'arousal'] = 'Medium 😐'
    convert_to_discrete(frame, 'valence')
    convert_to_discrete(frame, 'dominance')
    convert_to_discrete(frame, 'productivity')
    return frame


def convert_to_discrete(frame, arg1):
    # rename the value of the column "valence" where the value is 0 or 1 to "low" and 2 to "medium" and 3 or 4 to "high"
    frame[arg1] = frame[arg1].replace([3.0], 'Medium 😐')
    if arg1 == 'valence':
        frame[arg1] = frame[arg1].replace([1.0, 2.0], 'Low 😔')
        frame[arg1] = frame[arg1].replace([4.0, 5.0], 'High 😄')
    elif arg1 == 'dominance':
        frame[arg1] = frame[arg1].replace([1.0, 2.0], 'Low 😔🥱')
        frame[arg1] = frame[arg1].replace([4.0, 5.0], 'High 👨‍🎓')


def process(EDA, ACC, TEMP, popup):

    artifact_file = os.path.join(artifact_output_path, "artifact_detected.csv")
    ouput_path = os.path.join(artifact_output_path, "result.csv")
    classify_artifacts(classifierList, EDA, ACC, TEMP, artifact_file, ouput_path)
    data = detect_peak(ouput_path, artifact_file, thresh, offset, start_WT, end_WT)
    print(data['filtered_eda'].empty)
    data['timestamp'] = pd.to_datetime(data.index.values, utc=True).tz_convert('Europe/Berlin')
    data['timestamp'] = data['timestamp'] + data['timestamp'].apply(lambda x: x.utcoffset())
    popup['timestamp'] = pd.to_datetime(popup['timestamp'], utc=True)
    data['hours'] = data['timestamp'].dt.hour
    popup['hours'] = popup['timestamp'].dt.hour
    print(popup)
    df_merged = pd.merge(popup, data, on='timestamp', how='outer')
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'].values, utc=True)
    df_merged = df_merged.sort_values(by='timestamp')
    df_merged.to_csv('./temp/merged.csv')
    print(data['filtered_eda'].empty)
    df_popup = df_merged[
        ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity', 'notes', 'filtered_eda']]
    # drop the row if the column activity is nan
    df_popup = df_popup[df_popup['activity'].notna()]
    # plot df_popup in another fig
    datasrc = ColumnDataSource(df_popup)
    fig = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
                 title='EDA with Peaks marked as vertical lines', x_axis_label='Time', y_axis_label='μS',
                 sizing_mode='stretch_both')

    # Define the data source
    data_src = ColumnDataSource(df_merged)

    line_plot = fig.line(x='timestamp', y='filtered_eda', source=data_src)
    circle_plot = fig.circle(name='report', x='timestamp', y='filtered_eda', source=datasrc, fill_color="red",
                             size=9)

    line_hover = HoverTool(renderers=[line_plot],
                           tooltips=[("EDA", "@filtered_eda"), ("Timestamp", "@timestamp{%F}")],
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
    df_peak = df_peak.fillna(method='backfill').fillna(method='ffill').loc[~(df_peak['peaks_plot'] == 0)]
    for t, a in df_peak.iterrows():
        if a['arousal'] == 'Low':
            color = '#4DBD33'
        elif a['arousal'] == 'Medium':
            color = '#FF8C00'
        else:
            color = '#FF0000'
        fig.add_layout(Span(location=t, dimension='height', line_color=color, line_alpha=0.5, line_width=1))

    bokeh_pane.object = fig


def file_upload_handler(event):
    # Get the uploaded file
    file = event.new
    buffer = io.BytesIO(file)
    with zipfile.ZipFile(buffer) as zip_file:
        zip_file.extractall('./temp')


def start_process(event):
    EDA_df = pd.read_csv('./temp/EDA.csv')
    ACC_df = pd.read_csv('./temp/ACC.csv')
    TEMP_df = pd.read_csv('./temp/TEMP.csv')
    popup_df = popup_process()
    process(EDA_df, ACC_df, TEMP_df, popup_df)

button = pn.widgets.Button(name='Start Process', button_type='primary')
button.on_click(start_process)
fig = file_upload.param.watch(file_upload_handler, 'value')
# Create a Panel layout for the dashboard
layout = pn.Column("# Upload the Zip file of Empatica E4", file_upload, button, bokeh_pane, sizing_mode='stretch_both')
pn.extension()
layout.show()


