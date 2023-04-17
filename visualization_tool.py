import datetime
import os
import re
import shutil
from bokeh.models import Span
import io
import zipfile
import constant
import pandas as pd
from visualization_utils import classify_artifacts, create_directories_session_popup, create_directories_session_data, detect_peak, get_session, popup_process, process_data_popup, process_acc, process_hr, create_fig_line, save_EDAs_filtered, save_data_filtered
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


bokeh_pane_eda = pn.pane.Bokeh(visible=False, sizing_mode='stretch_both')
bokeh_pane_hr = pn.pane.Bokeh(visible=False, sizing_mode='stretch_both')
bokeh_pane_acc = pn.pane.Bokeh(visible=False, sizing_mode='stretch_both')

text_title_student = pn.widgets.StaticText()
text_title_day = pn.widgets.StaticText()
text_title_session = pn.widgets.StaticText()



file_name_student = None
current_session = None #Timestamp della sessione scelta
path_student = None #Path dello studente
path_days = None    #Path dei giorni di lavoro dello studente
path_sessions = None #Path delle sessoni di un giorno di lavoro
sessions = [] # Lista dei timestamp delle sessioni

pn.extension()


config_data = configparser.ConfigParser()
config_data.read("config.ini")
plot = config_data["PLOT"]



def process(date, session):
    global bokeh_pane_acc
    global bokeh_pane_eda
    global bokeh_pane_hr
    global progress_bar

    global plot
    
    #Check for missing signals in config
    signals = ['EDA', 'HR', 'ACC']
    for s in signals:
        if s not in plot.keys():
            plot[s] = '0'

    path_session = './temp/' + file_name_student + '/Sessions/' + date + '/' + session
    
    #x_range serve per muovere i grafici insieme sull'asse x
    x_range = None
    
    #EDA
    if int(plot['EDA']) == 1:
        bokeh_pane_eda.visible = True

        data = pd.read_csv(path_session + '/Data/data_eda_filtered.csv')
        df_merged = pd.read_csv(path_session + '/Data/df_merged_eda_filtered.csv')
        df_data = pd.read_csv(path_session + '/Data/df_data_eda_filtered.csv')
        df_popup = pd.read_csv(path_session + '/Data/df_popup_filtered.csv')


        df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], utc=True)

        df_data['timestamp'] = df_data['timestamp'].apply(lambda x: x.time())


        fig_eda = create_fig_line(df_data, 'timestamp', 'filtered_eda', 'EDA with Peaks marked as vertical lines', 'μS', 'EDA', df_popup)
        # Add the peak markers to the figure
        peak_height = data['filtered_eda'].max() * 1.15
        df_merged['peaks_plot'] = df_merged['peaks'] * peak_height
        df_peak = df_merged[['timestamp', 'peaks_plot', 'arousal']].set_index('timestamp')
        df_peak = df_peak.fillna(method='backfill').fillna(method='ffill').loc[~(df_peak['peaks_plot'] == 0)]
        df_peak.index = pd.to_datetime(df_peak.index.values)
        for t, a in df_peak.iterrows():
            timestamp = t.time()
            if a['arousal'] == 'Low 🧘‍♀':
                color = '#4DBD33'
            elif a['arousal'] == 'Medium 😐':
                color = '#FF8C00'
            else:
                color = '#FF0000'
            fig_eda.add_layout(Span(location=timestamp, dimension='height', line_color=color, line_alpha=0.5, line_width=1))
        
        if x_range is None:
            x_range = fig_eda.x_range
        
        fig_eda.x_range = x_range
        bokeh_pane_eda.object = fig_eda

    
    
    
    _, _, _, df_popup = get_session(path_session)
    df_popup['timestamp'] = pd.to_datetime(df_popup['timestamp'], utc=True)
    df_popup['timestamp'] = pd.to_datetime(df_popup['timestamp'].values, utc=True)
    df_popup['timestamp'] = df_popup['timestamp'].apply(lambda x: x.time())
    

    #ACC
    if int(plot['ACC']) == 1:
        
        bokeh_pane_acc.visible = True
        df_acc = pd.read_csv(path_session + '/Data/df_data_acc_filtered.csv')
        df_acc['timestamp'] = pd.to_datetime(df_acc['timestamp'], utc=True)
        df_acc['timestamp'] = df_acc['timestamp'].apply(lambda x: x.time())

        fig_acc = create_fig_line(df_acc, 'timestamp', 'acc_filter', 'Movement', 'Variation', 'MOV', df_popup)
        
        if x_range is None:
            x_range = fig_acc.x_range
        
        fig_acc.x_range = x_range
        
        bokeh_pane_acc.object = fig_acc
    
    #HR
    if int(plot['HR']) == 1:
        bokeh_pane_hr.visible = True
        df_hr = pd.read_csv(path_session + '/Data/df_data_hr_filtered.csv')
        df_hr['timestamp'] = pd.to_datetime(df_hr['timestamp'], utc=True)
        df_hr['timestamp'] = df_hr['timestamp'].apply(lambda x: x.time())


        fig_hr = create_fig_line(df_hr, 'timestamp', 'hr', 'Heart Rate', 'BPM', 'HR', df_popup)
        if x_range is None:
            x_range = fig_hr.x_range
        
        fig_hr.x_range = x_range
        
        bokeh_pane_hr.object = fig_hr
    progress_bar.visible = False

    print('Fine')

def file_upload_handler(event):
    # Get the uploaded file
    _file = event.new
    _buffer = io.BytesIO(_file)
    #Extract data and popup
    with zipfile.ZipFile(_buffer) as zip_file:
        global file_name_student
        #Get file name
        file_name_student = zip_file.namelist()[0].rsplit('/')[0]
        path_student = './temp/' + file_name_student

        global path_days
        path_days = path_student + '/Sessions'

        #Se esiste già la cartella, la elimino
        if os.path.exists(path_student):
            # Delete Folder code
            shutil.rmtree(path_student)

        for file in zip_file.namelist():
            if file.startswith(file_name_student + '/Data/') or file.startswith(file_name_student + '/Popup/'):
                zip_file.extract(member=file, path ='./temp/')

    create_directories_session_data(path_student)
    create_directories_session_popup(path_student)

    thresh.disabled = False
    offset.disabled = False
    start_WT.disabled = False
    end_WT.disabled = False
    button_student.disabled = False



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

    #Ricavare il giorno dalla sessione
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

    global current_session
    current_session = num_session_to_timestamp(num_session)
 
    global text_title_day, text_title_student
    text_title_day.value = 'Day: ' + day
    text_title_session.value = session

    process(day, current_session)


def num_session_to_timestamp(num_session):
    global sessions
    sorted_list = sorted(sessions)

    return sorted_list[num_session-1]

def create_select_sessions(event):
    global path_days
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

    #Attivazione dei parametri dell'EDA e della scelta delle sessioni
    global thresh, offset, start_WT, end_WT, button_session
    

    global text_title_student
    text_title_student.value = 'Student: ' + file_name_student
    save_data_filtered(path_days, thresh, offset, start_WT, end_WT)

    select.disabled = False
    button_session.disabled = False



#######                 #######
#######                 #######
#######     WIDGET      #######
#######                 #######
#######                 #######


#Inserimento del file zip
file_upload = FileInput(accept='.zip', sizing_mode='stretch_width')
fig = file_upload.param.watch(file_upload_handler, 'value')

#Button per confermare lo studente
button_student = pn.widgets.Button(name='Confirm student', button_type='primary', disabled = True, sizing_mode='stretch_width')
button_student.on_click(create_select_sessions)

#Progress Bar
progress_bar = pn.indicators.Progress(name = 'Progress', visible=False, active=True, sizing_mode='stretch_width')

#Selezione della sessione
select = pn.widgets.Select(name='Select Session', options=sessions, disabled = True, sizing_mode='stretch_width')


#Parametri EDA
thresh = pn.widgets.TextInput(name='Peak width', placeholder='default .02', value='.02', disabled = True, sizing_mode='stretch_width')
offset = pn.widgets.TextInput(name='Peak start time', placeholder='default 1', value='1', disabled = True, sizing_mode='stretch_width')
start_WT = pn.widgets.TextInput(name='Peak end time', placeholder='default 4', value='4', disabled = True, sizing_mode='stretch_width')
end_WT = pn.widgets.TextInput(name='Minimum peak amplitude', placeholder='default 4', value='4', disabled = True, sizing_mode='stretch_width')
params_col = pn.Column(offset, thresh, start_WT, end_WT, visible = False, sizing_mode='stretch_width')

# Se EDA è attivato nel file di configurazione, vengono visualizzati i parametri per i picchi
if int(plot['EDA']) == 1:
    params_col.visible = True

#Button per confermare la sessione
button_session = pn.widgets.Button(name='Start Process', button_type='primary', disabled = True, sizing_mode='stretch_width')
button_session.on_click(start_process)

#Template
template = pn.template.FastGridTemplate(
    site="EmoVizPhy", title = '',
    sidebar=[file_upload, params_col, button_student, select, button_session, progress_bar],
    theme_toggle = False
)

#Header
title = pn.Row(pn.layout.HSpacer(), text_title_student, text_title_day, text_title_session)  
template.header.append(title)

#Main
#Il  numero di panel mostrati è uguale al numero di segnali da mostrare. Se ad esempio, nel file config EDA è 
#disattivato, allora bisogna rimuovere il suo panel
show_bokeh_pane = []
if int(plot['EDA']) == 1:
    show_bokeh_pane.append(bokeh_pane_eda)
if int(plot['HR']) == 1:
    show_bokeh_pane.append(bokeh_pane_hr)
if int(plot['ACC']) == 1:
    show_bokeh_pane.append(bokeh_pane_acc)

size = 2
for i in range(len(show_bokeh_pane)):
    #12 è il massimo
    template.main[(i*size):(i*size)+size, :] = show_bokeh_pane[i]



print("Reach the application at http://localhost:20000")

template.show(port = 20000)