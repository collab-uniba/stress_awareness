import datetime
import os
import re
import shutil
from bokeh.models import Span
import pandas as pd
from visualization_utils import create_directories_session_popup, create_directories_session_data, get_popup, create_fig_line, save_data_filtered
import panel as pn
import configparser
from scipy.stats import rankdata
import shutil
from tkinter import Tk
from tkinter.filedialog import askdirectory


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

selected_path_directory = None

#Parametri EDA
thresh = pn.widgets.TextInput(name='Peak width', placeholder='default .02', value='.02', disabled = True, sizing_mode='stretch_width')
offset = pn.widgets.TextInput(name='Peak start time', placeholder='default 1', value='1', disabled = True, sizing_mode='stretch_width')
start_WT = pn.widgets.TextInput(name='Peak end time', placeholder='default 4', value='4', disabled = True, sizing_mode='stretch_width')
end_WT = pn.widgets.TextInput(name='Minimum peak amplitude', placeholder='default 4', value='4', disabled = True, sizing_mode='stretch_width')
params_col = pn.Column(offset, thresh, start_WT, end_WT, visible = False, sizing_mode='stretch_width')

#Selezione della directory
dir_input_btn = pn.widgets.Button(name="Select Data Directory", button_type='primary', sizing_mode='stretch_width', height=50)
dir_input_btn.on_click(lambda x: select_directory())

file_name_student = None
current_session = None #Timestamp della sessione scelta
path_student = None #Path dello studente
path_days = None    #Path dei giorni di lavoro dello studente
path_sessions = None #Path delle sessioni di un giorno di lavoro
sessions = [] # Lista dei timestamp delle sessioni

pn.extension()


config_data = configparser.ConfigParser()
config_data.read("config.ini")
plot = config_data["PLOT"]












def select_directory():
    #Questo metodo permette di selezionare la cartella
    global selected_path_directory
    global text_title_student
    
    root = Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    dirname = askdirectory()
    if dirname:
        selected_path_directory = dirname

    prepare_files(selected_path_directory)
    
    global file_name_student
    text_title_student.value = 'Directory ' + file_name_student + ' selected'
    dir_input_btn.background = '#00A170'

    dir_input_btn.aspect_ratio

    reset_widgets()
    
    
    

def reset_widgets():
    global button_visualize
    global bokeh_pane_eda, bokeh_pane_acc, bokeh_pane_hr
    global select
    button_visualize.disabled = True
    bokeh_pane_eda.visible = False
    bokeh_pane_acc.visible = False
    bokeh_pane_hr.visible = False
    
    global text_title_day, text_title_session
    text_title_day.value = ''
    text_title_session.value = ''

    select.disabled = True

    


def prepare_files(path):
    #Questo metodo copia e prepara i file nella cartella temp
    global file_name_student
    #Get file directory
    file_name_student = os.path.basename(path)

    path_student = './temp/' + file_name_student

    global path_days
    path_days = path_student + '/Sessions'

    #Se esiste già la cartella in temp, la elimino
    if os.path.exists('./temp/'):
        # Delete Folders
        shutil.rmtree('./temp/')

    os.mkdir('./temp/')
    os.mkdir(path_student)
    
    shutil.copytree(path + '/Data', path_student + '/Data')
    shutil.copytree(path + '/Popup', path_student + '/Popup')
    
    create_directories_session_data(path_student)
    create_directories_session_popup(path_student)

    thresh.disabled = False
    offset.disabled = False
    start_WT.disabled = False
    end_WT.disabled = False
    button_analyse.disabled = False



def visualize_session(date, session):
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
    popup = get_popup(path_session , date)
    
    
    
    #EDA
    if int(plot['EDA']) == 1:
        bokeh_pane_eda.visible = True

        data = pd.read_csv(path_session + '/Data/data_eda_filtered.csv')
        data['time'] = pd.to_datetime(data['timestamp']).dt.time
        data = data[['time', 'filtered_eda', 'peaks']]
        
        fig_eda = create_fig_line(data, 'time', 'filtered_eda', 'Electrodermal Activity', 'μS', 'EDA', popup)
        
        # Add the peak markers to the figure
        peak_height = data['filtered_eda'].max() * 1.15
        data['peaks_plot'] = data['peaks'] * peak_height
        time_peaks = data[data['peaks_plot'] != 0]['time']
        
        for t in time_peaks:
            color = '#4DBD33'
            fig_eda.add_layout(Span(location=t, dimension='height', line_color=color, line_alpha=0.5, line_width=1))
        
        if x_range is None:
            x_range = fig_eda.x_range
        
        fig_eda.x_range = x_range
        bokeh_pane_eda.object = fig_eda
    
    
    

    
    #ACC
    if int(plot['ACC']) == 1:
        
        bokeh_pane_acc.visible = True
        df_acc = pd.read_csv(path_session + '/Data/df_data_acc_filtered.csv')
        
        df_acc['time'] = pd.to_datetime(df_acc['timestamp'], utc=True).dt.time
        df_acc = df_acc[['time', 'acc_filter']]
        
        fig_acc = create_fig_line(df_acc, 'time', 'acc_filter', 'Movement', 'Variation', 'MOV', popup)
        
        if x_range is None:
            x_range = fig_acc.x_range
        
        fig_acc.x_range = x_range
        
        bokeh_pane_acc.object = fig_acc
    
    #HR
    if int(plot['HR']) == 1:
        bokeh_pane_hr.visible = True
        df_hr = pd.read_csv(path_session + '/Data/df_data_hr_filtered.csv')
        
        df_hr['time'] = pd.to_datetime(df_hr['timestamp'], utc=True).dt.time
        df_hr = df_hr[['time', 'hr']]

        fig_hr = create_fig_line(df_hr, 'time', 'hr', 'Heart Rate', 'BPM', 'HR', popup)
        if x_range is None:
            x_range = fig_hr.x_range
        
        fig_hr.x_range = x_range
        
        bokeh_pane_hr.object = fig_hr
    
    progress_bar.visible = False

    print('Fine')


def prepare_sessions(event):
    #Questo metodo ricava il giorno e la sessione dal valore della select
    global progress_bar
    progress_bar.visible = True
    
    global select
    groups = select.groups
    session = select.value
    
    day = None

    #Ricavare il giorno dalla stringa "Session #: HH:MM:SS"
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
    
    visualize_session(day, current_session)
    

def num_session_to_timestamp(num_session):
    global sessions
    sorted_list = sorted(sessions)

    return sorted_list[num_session-1]

def create_select_sessions(event):




    global button_analyse
    global offset
    global thresh
    global start_WT
    global end_WT
    global dir_input_btn
    #Disattivo i bottoni

    dir_input_btn.disabled = True
    button_analyse.disabled = True
    offset.disabled = True
    thresh.disabled = True
    start_WT.disabled = True 
    end_WT.disabled = True
    #Questo metodo converte i timestamp delle sessioni nella stringa "Session #: HH:MM:SS"
    global path_days
    days = os.listdir(path_days)
    
    # Dizionario con key: giorno    value: lista di stringhe "Session #: HH:MM:SS"
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
    global text_title_student
    text_title_student.value = 'Analysing ' + file_name_student
    save_data_filtered(path_days, thresh, offset, start_WT, end_WT)

    #Visualizza la prima sessione
    prepare_sessions(event)

    dir_input_btn.disabled = False
    button_analyse.disabled = False
    select.disabled = False
    button_visualize.disabled = False
    offset.disabled = False
    thresh.disabled = False
    start_WT.disabled = False 
    end_WT.disabled = False


#######                 #######
#######                 #######
#######     WIDGET      #######
#######                 #######
#######                 #######

#Button per confermare lo studente
button_analyse = pn.widgets.Button(name='Analyse biometrics', button_type='primary', disabled = True, sizing_mode='stretch_width')
button_analyse.on_click(create_select_sessions)

#Progress Bar
progress_bar = pn.indicators.Progress(name = 'Progress', visible=False, active=True, sizing_mode='stretch_width')

#Selezione della sessione
select = pn.widgets.Select(name='Select Session', options=sessions, disabled = True, sizing_mode='stretch_width')


#Button per visualizzare la sessione
button_visualize = pn.widgets.Button(name='Visualize session', button_type='primary', disabled = True, sizing_mode='stretch_width')
button_visualize.on_click(prepare_sessions)


# Se EDA è attivato nel file di configurazione, vengono visualizzati i parametri per i picchi
if int(plot['EDA']) == 1:
    params_col.visible = True

#Template
template = pn.template.FastGridTemplate(
    title = 'EmoVizPhy',
    sidebar=[dir_input_btn, 
             params_col,
             button_analyse,
             select, 
             button_visualize, 
             progress_bar],
    theme_toggle = False,
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
#template.show()
template.show(port = 20000)