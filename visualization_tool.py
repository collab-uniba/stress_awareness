import datetime
import os
import re
from bokeh.models import Span
import io
import zipfile
import constant
import pandas as pd
from visualization_utils import classify_artifacts, create_directories_session_popup, create_directories_session_data, detect_peak, popup_process, process_data_popup, process_acc, process_hr, create_fig_line
import panel as pn
from panel.widgets import FileInput
import configparser
from scipy.stats import rankdata

'''
 Please note that this script use scripts released by Taylor et al. that you can find here: https://github.com/MITMediaLabAffectiveComputing/eda-explorer
 
  Taylor, Sara et al. “Automatic identification of artifacts in electrodermal activity data.” 
  Annual International Conference of the IEEE Engineering in Medicine and Biology Society. 
  IEEE Engineering in Medicine and Biology Society. 
  Annual International Conference vol. 2015 (2015): 1934-7. 
  doi:10.1109/EMBC.2015.7318762

'''

file_upload = FileInput(accept='.zip')
thresh = pn.widgets.TextInput(name='Peak width', placeholder='default .02', value='.02')
offset = pn.widgets.TextInput(name='Peak start time', placeholder='default 1', value='1')
start_WT = pn.widgets.TextInput(name='Peak end time', placeholder='default 4', value='4')
end_WT = pn.widgets.TextInput(name='Minimum peak amplitude', placeholder='default 4', value='4')
bokeh_pane_eda = pn.pane.Bokeh()
bokeh_pane_acc = pn.pane.Bokeh()
bokeh_pane_hr = pn.pane.Bokeh()



file_zip_name_student = None 
current_session = None #Timestamp della sessione scelta
path_days = None    #Path che porta ai giorni
path_sessions = None #Path che porta alle sessioni
sessions = [] # Lista dei timestamp delle sessioni

pn.extension()

def process(date, session):
    global bokeh_pane_acc
    global bokeh_pane_eda
    global bokeh_pane_hr
    global progress_bar

    config_data = configparser.ConfigParser()
    config_data.read("config.ini")
    plot = config_data["PLOT"]
    
    #Check for missing signals in config
    signals = ['EDA', 'HR', 'ACC']
    for s in signals:
        if s not in plot.keys():
            plot[s] = '0'

    path_session = './temp/' + file_zip_name_student + '/Sessions/' + date + '/' + session
    

    progress_bar.value = 20
    #EDA
    if int(plot['EDA']) == 1:

        EDA, ACC, TEMP, popup = get_session(path_session)
        artifact_file = os.path.join(constant.artifact_output_path, "artifact_detected.csv")

        
        progress_bar.value = 30

        output_file_path = os.path.join(constant.artifact_output_path, "result.csv")
        classify_artifacts(EDA, ACC, TEMP, artifact_file, output_file_path)

        progress_bar.value = 40
        data = detect_peak(output_file_path, artifact_file, thresh, offset, start_WT, end_WT)
        
        progress_bar.value = 50
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
    
    progress_bar.value = 70
    
    EDA, ACC, TEMP, df_popup = get_session(path_session)
    df_popup['timestamp'] = pd.to_datetime(df_popup['timestamp'], utc=True)
    df_popup['timestamp'] = pd.to_datetime(df_popup['timestamp'].values, utc=True)
    df_popup['timestamp'] = df_popup['timestamp'].apply(lambda x: x.time())
    
    #ACC
    if int(plot['ACC']) == 1:
        df_acc  = process_acc(path_session)
        bokeh_pane_acc.object = create_fig_line(df_acc, 'timestamp', 'acc_filter', 'Movement', '', 'ACC', df_popup)
    
    progress_bar.value = 99
    
    #HR
    if int(plot['HR']) == 1:
        df_hr = process_hr(path_session)   
        bokeh_pane_hr.object = create_fig_line(df_hr, 'timestamp', 'hr', 'Heart Rate', 'BPM', 'HR', df_popup)
    
    progress_bar.visible = False

    print('Fine')

def file_upload_handler(event):
    # Get the uploaded file
    _file = event.new
    _buffer = io.BytesIO(_file)
    #Get file zip name
    global file_zip_name_student
    file_zip_name_student = file_upload.filename.rsplit('.', 1)[0]
    #Extract data and popup
    with zipfile.ZipFile(_buffer) as zip_file:
        for file in zip_file.namelist():
            #temp_eda_timestamp = zip_file.extract(member=file, path ='./temp/')
            if file.startswith(file_zip_name_student + '/Data/') or file.startswith(file_zip_name_student + '/Popup/'):
                zip_file.extract(member=file, path ='./temp/')


    dir_path = r'.\\temp\\' + file_zip_name_student
    
    create_directories_session_data(dir_path)
    create_directories_session_popup(dir_path)


    
def get_session(path_session):    
    EDA_df = pd.read_csv(path_session + '/Data/' + 'EDA.csv')
    ACC_df = pd.read_csv(path_session + '/Data/' + 'ACC.csv')
    TEMP_df = pd.read_csv(path_session + '/Data/' + 'TEMP.csv')
    popup_df = popup_process(path_session + '/Popup/' + 'popup.csv') 

    return EDA_df, ACC_df, TEMP_df, popup_df



def start_process(event):
    global bokeh_pane_acc
    global bokeh_pane_eda
    global bokeh_pane_hr
    global progress_bar

    progress_bar.visible = True
    global select
    
    groups = select.groups
    session = select.value
    day = None

    for key, values in groups.items():
        if str(session) in values:
            day = key
            break
    
    global path_sessions
    path_sessions = path_days +'/' + day

    global sessions
    sessions = os.listdir(path_sessions)

    #Esempio di session: 'Session 2: 12:13:49'
    num_session = int(re.search(r'\d+', session).group())

    global template
    

    global current_session
    current_session = num_session_to_timestamp(num_session)
 
    # TODO controllare perché non cambia il titolo
    template.title = file_zip_name_student + '    Day: ' + day + '   ' + session
    process(day, current_session)


def num_session_to_timestamp(num_session):
    global sessions
    sorted_list = sorted(sessions)

    return sorted_list[num_session-1]

def create_select_sessions(event):
    global path_days
    path_days = './temp/' + file_zip_name_student + '/Sessions'
    days = os.listdir(path_days)
    
    # Dizionario con key: giorno    value: lista di sessioni
    groups = {}
    for d in days:
        sessions = os.listdir(path_days + '/' + str(d))
        #Converto i timestamp delle sessioni in numero della sessione nella giornata
        dt_objects_list = [datetime.datetime.fromtimestamp(int(t)) for t in sessions]
        dt_objects_list = [datetime.datetime.strftime(t, '%H:%M:%S') for t in dt_objects_list]
        
        num_sessions = rankdata(sessions).astype(int)
        string_sessions = ['Session ' + str(n) + ': ' + s for n, s in zip(num_sessions, dt_objects_list)]
        
        groups[d] = string_sessions
    
    global select
    select.groups = groups





progress_bar = pn.indicators.Progress(name = 'Progress', visible=False, active=True, sizing_mode='stretch_width')

select = pn.widgets.Select(name='Select Session', options=sessions)
button_student = pn.widgets.Button(name='Confirm student', button_type='primary')
button_student.on_click(create_select_sessions)

button_session = pn.widgets.Button(name='Start Process', button_type='primary')
button_session.on_click(start_process)
fig = file_upload.param.watch(file_upload_handler, 'value')
# Create a Panel layout for the dashboard
params_row = pn.Column(offset, thresh, start_WT, end_WT)
'''
layout = pn.Column("# Upload the Zip file of Empatica E4", 
                   file_upload, params_row, 
                   button_student, 
                   select, button_session,
                   progress_bar,
                   bokeh_pane_eda,bokeh_pane_hr,  bokeh_pane_acc,
                   sizing_mode='stretch_width')

layout.show()
'''

title = ''
col = pn.Column(bokeh_pane_acc, bokeh_pane_hr, sizing_mode='stretch_width')
template = pn.template.FastGridTemplate(
    site="EmoVizPhy", title=title,
    sidebar=[file_upload, button_student, params_row, select, button_session, progress_bar],
    #main = [col]
)
#12 è il massimo
template.main[:2, :12] = bokeh_pane_eda
template.main[2:4, :12] = bokeh_pane_hr
template.main[4:6, :12] = bokeh_pane_acc

#template.main[:3, 6:] = pn.pane.HoloViews(bokeh_pane_hr, sizing_mode="stretch_both")

template.show()