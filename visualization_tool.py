import os
from bokeh.models import Span
import io
import zipfile
import constant
import pandas as pd
from visualization_utils import classify_artifacts, detect_peak, popup_process, process_data_popup, process_acc, process_hr, create_fig_line
import panel as pn
from panel.widgets import FileInput
import configparser

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
bokeh_pane_eda = pn.pane.Bokeh()
bokeh_pane_acc = pn.pane.Bokeh()
bokeh_pane_hr = pn.pane.Bokeh()

def process(EDA, ACC, TEMP, popup):
    config_data = configparser.ConfigParser()
    config_data.read("config.ini")
    plot = config_data["PLOT"]
    
    #Check for missing signals in config
    signals = ['EDA', 'HR', 'ACC']
    for s in signals:
        if s not in plot.keys():
            plot[s] = '0'



    #EDA
    if int(plot['EDA']) == 1:
        artifact_file = os.path.join(constant.artifact_output_path, "artifact_detected.csv")
        output_file_path = os.path.join(constant.artifact_output_path, "result.csv")
        classify_artifacts(EDA, ACC, TEMP, artifact_file, output_file_path)
        data = detect_peak(output_file_path, artifact_file, thresh, offset, start_WT, end_WT)
        
        df_merged = process_data_popup(data, popup)
        df_merged['timestamp'] = df_merged['timestamp'].apply(lambda x: x.time())

        df_EDA = df_merged[
            ['timestamp', 'activity', 'status_popup', 'valence', 'arousal', 'dominance', 'productivity', 'notes', 'filtered_eda']]
       
        df_data = df_EDA[df_EDA['status_popup'].isna()]
        df_data = df_data[['timestamp', 'filtered_eda']]
        df_data.reset_index(inplace=True, drop=True)
        
        df_popup = df_EDA[df_EDA['status_popup'] == 'POPUP_CLOSED']
        
        
        fig_eda = create_fig_line(df_data, 'timestamp', 'filtered_eda', 'EDA with Peaks marked as vertical lines', 'μS', 'EDA', df_popup)

        # Add the peak markers to the figure
        peak_height = data['filtered_eda'].max() * 1.15
        df_merged['peaks_plot'] = df_merged['peaks'] * peak_height
        df_peak = df_merged[['timestamp', 'peaks_plot', 'arousal']].set_index('timestamp')
        df_peak = df_peak.fillna(method='backfill').fillna(method='ffill').loc[~(df_peak['peaks_plot'] == 0)]
        
        for t, a in df_peak.iterrows():
            if a['arousal'] == 'Low 🧘‍♀':
                color = '#4DBD33'
            elif a['arousal'] == 'Medium 😐':
                color = '#FF8C00'
            else:
                color = '#FF0000'
            fig_eda.add_layout(Span(location=t, dimension='height', line_color=color, line_alpha=0.5, line_width=1))
        bokeh_pane_eda.object = fig_eda
    
    #ACC
    if int(plot['ACC']) == 1:
        df_acc  = process_acc()
        bokeh_pane_acc.object = create_fig_line(df_acc, 'timestamp', 'acc_filter', 'Movement', '', 'ACC')
    
    #HR
    if int(plot['HR']) == 1:
        df_hr = process_hr()   
        bokeh_pane_hr.object = create_fig_line(df_hr, 'timestamp', 'hr', 'Heart Rate', 'BPM', 'HR')


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
layout = pn.Column("# Upload the Zip file of Empatica E4", file_upload,params_row, button, bokeh_pane_eda,bokeh_pane_hr,  bokeh_pane_acc, sizing_mode='stretch_both')
pn.extension()
layout.show()


