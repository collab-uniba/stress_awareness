import os
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span, HoverTool
import io
import zipfile
import constant
import pandas as pd
from visualization_utils import classify_artifacts, detect_peak, popup_process, convert_to_datetime, process_acc_hr
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

file_upload = FileInput()
thresh = pn.widgets.TextInput(name='Peak width', placeholder='default .02', value='.02')
offset = pn.widgets.TextInput(name='Peak start time', placeholder='default 1', value='1')
start_WT = pn.widgets.TextInput(name='Peak end time', placeholder='default 4', value='4')
end_WT = pn.widgets.TextInput(name='Minimum peak amplitude', placeholder='default 4', value='4')
bokeh_pane = pn.pane.Bokeh()
bokeh_pane_acc = pn.pane.Bokeh()
bokeh_pane_hr = pn.pane.Bokeh()


def process(EDA, ACC, TEMP, popup):
    artifact_file = os.path.join(constant.artifact_output_path, "artifact_detected.csv")
    output_file_path = os.path.join(constant.artifact_output_path, "result.csv")
    classify_artifacts(EDA, ACC, TEMP, artifact_file, output_file_path)
    data = detect_peak(output_file_path, artifact_file, thresh, offset, start_WT, end_WT)
    df_merged = convert_to_datetime(data, popup)
    df_popup = df_merged[
        ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity', 'notes', 'filtered_eda']]
    df_popup = df_popup[df_popup['activity'].notna()]
    df_acc, df_hr = process_acc_hr()
    datasrc = ColumnDataSource(df_popup)

    fig = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
                 title='EDA with Peaks marked as vertical lines', x_axis_label='Time', y_axis_label='μS',
                 sizing_mode='stretch_both')

    fig_hr = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
                    title='Heart Rate', x_axis_label='Time', y_axis_label='BPM',
                    sizing_mode='stretch_both')

    fig_ACC = figure(x_axis_type='datetime', plot_width=1500, plot_height=400,
                 title='Movement', x_axis_label='Time',
                 sizing_mode='stretch_both')

    data_src = ColumnDataSource(df_merged)
    data_src_acc = ColumnDataSource(df_acc)
    data_src_hr = ColumnDataSource(df_hr)

    line_plot_hr = fig_hr.line(x='timestamp', y='hr', source=data_src_hr)
    line_plot_acc = fig_ACC.line(x='timestamp', y='acc_filter', source=data_src_acc)
    line_plot = fig.line(x='timestamp', y='filtered_eda', source=data_src)
    circle_plot = fig.circle(name='report', x='timestamp', y='filtered_eda', source=datasrc, fill_color="yellow",
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
    bokeh_pane_acc.object = fig_ACC
    bokeh_pane_hr.object = fig_hr


def file_upload_handler(event):
    # Get the uploaded file
    _file = event.new
    _buffer = io.BytesIO(_file)
    with zipfile.ZipFile(_buffer) as zip_file:
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
params_row = pn.Row(offset, thresh, start_WT, end_WT)
layout = pn.Column("# Upload the Zip file of Empatica E4", file_upload,params_row, button, bokeh_pane,bokeh_pane_hr,  bokeh_pane_acc, sizing_mode='stretch_both')
pn.extension()
layout.show()


