import configparser
import datetime
import os
import zipfile
import numpy as np
import constant
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure

from signalPreprocess import EDA_Artifact_Detection_Script as eda_artifact
from signalPreprocess import EDA_Peak_Detection_Script as eda_peak
from signalPreprocess import ACC_HR_Filtering as accelerometer

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


def popup_process(path_popup):
    frame = pd.read_csv(path_popup)


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

def process_acc(path_session):
    acc, timestamp_0 = accelerometer.load_acc(path_session)
    acc_filter = accelerometer.empatica_filter(acc)
    date, time = calculate_date_time(timestamp_0,1,len(acc_filter))
    # create a df with the filtered acc data and date and time
    df_acc = pd.DataFrame(acc_filter, columns=['acc_filter'])
    df_acc['time'] = time
    df_acc['date'] = date
    df_acc['timestamp'] = pd.to_datetime(df_acc['date'] + ' ' + df_acc['time'])
    return df_acc

def process_hr(path_session):
    hr, timestamp_0_hr = accelerometer.load_hr(path_session)
    date, time = calculate_date_time(timestamp_0_hr, 1, len(hr))
    df_hr = pd.DataFrame(hr, columns=['hr'])
    df_hr['time'] = time
    df_hr['date'] = date
    df_hr['timestamp'] = pd.to_datetime(df_hr['date'] + ' ' + df_hr['time'])

    return df_hr

def get_session(path_session):    
    EDA_df = pd.read_csv(path_session + '/Data/' + 'EDA.csv')
    ACC_df = pd.read_csv(path_session + '/Data/' + 'ACC.csv')
    TEMP_df = pd.read_csv(path_session + '/Data/' + 'TEMP.csv')
    popup_df = popup_process(path_session + '/Popup/' + 'popup.csv') 

    return EDA_df, ACC_df, TEMP_df, popup_df


def save_EDAs_filtered(path_days, thresh, offset, start_WT, end_WT):
    days = os.listdir(path_days)
    for d in days:
        path_sessions = path_days + '/' + d
        sessions = os.listdir(path_sessions)
        for s in sessions:
            path_session = path_days + '/' + d + '/' + s
            EDA, ACC, TEMP, popup = get_session(path_session + '/')
            artifact_file = os.path.join(constant.artifact_output_path, "artifact_detected.csv")

            output_file_path = os.path.join(constant.artifact_output_path, "result.csv")
            classify_artifacts(EDA, ACC, TEMP, artifact_file, output_file_path)

            data = detect_peak(output_file_path, artifact_file, thresh, offset, start_WT, end_WT)
            data['time'] = pd.to_datetime(data['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            
            df_merged = process_data_popup(data, popup)
            df_merged['time'] = pd.to_datetime(df_merged['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            data['time'] = pd.to_datetime(data['time']).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            
            df_EDA = df_merged[
                ['time', 'activity', 'status_popup', 'valence', 'arousal', 'dominance', 'productivity', 'notes', 'filtered_eda']]
            
            df_data = df_EDA[df_EDA['status_popup'].isna()]
            df_data = df_data[['time', 'filtered_eda']]
            df_data.reset_index(inplace=True, drop=True)
            
            df_popup = df_EDA[df_EDA['status_popup'] == 'POPUP_CLOSED']
            df_popup.reset_index(inplace=True, drop=True)
            

            data.to_csv(path_session + '/Data/data_eda_filtered.csv', index=False)
            df_merged.to_csv(path_session + '/Data/df_merged_eda_filtered.csv', index=False)
            df_data.to_csv(path_session + '/Data/df_data_eda_filtered.csv', index=False)
            df_popup.to_csv(path_session + '/Data/df_popup_filtered.csv', index=False)






def save_HRs_filtered(path_days):
    days = os.listdir(path_days)
    for d in days:
        path_sessions = path_days + '/' + d
        sessions = os.listdir(path_sessions)
        for s in sessions:
            path_session = path_days + '/' + d + '/' + s
            df_hr = process_hr(path_session)  
            df_hr.to_csv(path_session + '/Data/df_data_hr_filtered.csv', index=False)


def save_ACCs_filtered(path_days):
    days = os.listdir(path_days)
    for d in days:
        path_sessions = path_days + '/' + d
        sessions = os.listdir(path_sessions)
        for s in sessions:
            path_session = path_days + '/' + d + '/' + s
            df_acc  = process_acc(path_session)
            
            #Empatica suggerisce di rimuovere i primi 10 secondi
            df_acc = df_acc[10:]
            df_acc.reset_index(inplace=True, drop=True)
            df_acc.to_csv(path_session + '/Data/df_data_acc_filtered.csv', index=False)



def save_data_filtered(path_days, thresh, offset, start_WT, end_WT):
    config_data = configparser.ConfigParser()
    config_data.read("config.ini")
    plot = config_data["PLOT"]
    
    if int(plot['EDA']) == 1:
        save_EDAs_filtered(path_days, thresh, offset, start_WT, end_WT)
        
    if int(plot['HR']) == 1:
        save_HRs_filtered(path_days)
        
    if int(plot['ACC']) == 1:
        save_ACCs_filtered(path_days)
        




def create_fig_line(df_sign, x, y, title, y_axis_label, sign, df_popup):
    fig_sign = figure(x_axis_type='datetime', plot_height=400,
                    title=title, x_axis_label='Time', y_axis_label=y_axis_label,
                    sizing_mode='stretch_both', tools = ['pan', 'xpan', 'box_zoom' ,'reset', 'save'])
    #Rimozione della griglia dal background
    fig_sign.xgrid.grid_line_color = None
    fig_sign.ygrid.grid_line_color = None
    
    data_src_sign = ColumnDataSource(df_sign)
    line_plot_sign = fig_sign.line(x=x, y=y, source=data_src_sign)
    

    line_hover = HoverTool(renderers=[line_plot_sign],
                            tooltips=[(sign, "@"+y), ("Time", "@timestamp{%H:%M:%S}")],
                            formatters={'@timestamp': 'datetime'})
    fig_sign.add_tools(line_hover)

    
    #Mean
    mean = df_sign.loc[:, y].mean()
    fig_sign.add_layout(Span(location=mean, dimension='width', line_color="red", line_alpha=0.5, line_width=1, line_dash='dashed'))
    

    # Il seguente controllo è necessario per assegnare ai popup dei timestamp esistenti nei segnali (per la visualizzazione)
    # Assegno i valori dei segnali ai popup nei relativi timestamp
    df_temp = df_sign.copy()
    df_popup_copy = df_popup.copy()
 
    #Assegnazione dei valori y ai popup
    df_popup_copy[y] = None
    for i in range(df_popup_copy.shape[0]):
        time = df_popup_copy.loc[i,x]
        
        # EDA ha il timestamp in un altro formato
        if sign == 'EDA':
            time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f').time()
        
        
        temp = df_temp[df_temp[x] == time]
        if not temp.empty:
            temp.reset_index(inplace=True, drop=True)
           
            # Assegno il valore del segnale
            df_popup_copy.loc[i, y] = temp.loc[0,y]
            # Assegno il valore del timestamp
            df_popup_copy.loc[i, x] = temp.loc[0,x]
    
    #Se ci sono popup con time che non sono presenti nei segnali, non vengono considerati
    df_popup_copy = df_popup_copy[df_popup_copy[y].notna()]
    
    df_popup_copy[df_popup_copy['notes'].isna()]['notes'] = ''    
    
    datasrc = ColumnDataSource(df_popup_copy)
    circle_plot = fig_sign.circle(name='report', x=x, y=y, source=datasrc, fill_color="yellow",
                               size=9)
    circle_hover = HoverTool(renderers=[circle_plot],
                               tooltips=[("Activity", "@activity"), ("Valence", "@valence"), ("Arousal", "@arousal"),
                                        ("Dominance", "@dominance"), ("Productivity", "@productivity"),
                                        ("Notes", "@notes"), ("Time", "@time{%H:%M:%S}"), (sign, "@"+y)],
                                formatters={'@time': 'datetime'})
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



def create_directories_session_data(dir_path):
    #get all zip file names (sessions)
    zip_files_session = []
    for file in os.listdir(dir_path  + '/Data'):
        if file.endswith(".zip"):
            zip_files_session.append(file)
    
    for session_name in zip_files_session:
        timestamp_session = int(session_name.rsplit('_')[0])

        date_time_session = datetime.datetime.fromtimestamp(timestamp_session)
        dir_day = dir_path + '/Sessions/' + datetime.datetime.strftime(date_time_session, format='%d-%m-%Y')
          
        #time = str(date_time_session.time()).replace(':', '')
        # Lasciare il timestamp come nome delle cartelle. è necessario per capire quando è stato fatto un popup.
        # Ad esempio, se una sessione inizia alle 23.50 e un popup viene inserito alle 2.00 del giorno dopo,
        # potrebbe risultare difficile capire che il popup appartiene alla sessione del giorno prima.
        dir_session = dir_day + '/' + str(timestamp_session)
        if not os.path.exists(dir_session):
            os.makedirs(dir_session)
        
        dir_data_session = dir_session + '/' + 'Data/'
        
        with zipfile.ZipFile(dir_path + '/Data/' + session_name, "r") as zip_ref:
            zip_ref.extractall(dir_data_session)


def create_directories_session_popup(dir_path):
    # Ad ogni inserimento di popup viene creato un file che contiene tutti i popup, non solo quelli nuovi
    # In questo metodo si considera solo l'ultimo file

    #get all popup file names (sessions)
    popup_files_session = []
    dir_popup = dir_path  + '/Popup'
    for file in os.listdir(dir_popup):
        if file.startswith("data"):
            #Replace è necessario per cercare il massimo nella stringa (per l'ultimo popup)
            popup_files_session.append(file.replace('data.csv', 'data (0).csv'))
    # This is the last popup
    popup_file_session_name = max(popup_files_session).replace('data (0).csv', 'data.csv')
    
    all_popup = pd.read_csv(dir_popup + '/' + popup_file_session_name, 
                            names=['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity',
                                'status_popup', 'notes'])
    all_popup = all_popup[all_popup['status_popup'] == 'POPUP_CLOSED']
    all_popup.reset_index(inplace=True, drop=True)
    

    

    # Salvataggio dei popup nelle relative sessioni
    temp_df = all_popup.copy()
    temp_df['day'] = None
    temp_df['session'] = None
    # Assegnazione dei giorni di lavoro e delle sessioni ai popup
    for i in range(all_popup.shape[0]):
        date, session = get_date_session_popup(all_popup.loc[i, 'timestamp'], dir_path + '/Sessions')   
        temp_df.loc[i, 'day'] = date
        temp_df.loc[i, 'session'] = session

    
    # Rimozione dei popup senza una sessione
    temp_df = temp_df[temp_df['session'].notnull()]
    temp_df.reset_index(inplace=True, drop=True)
    
    # Salvataggio dei popup
    sessions = set(temp_df['session'].values)
    for s in sessions:
        popup_session = temp_df[temp_df['session'] == s]
        popup_session.reset_index(inplace=True, drop=True)
        day = popup_session.loc[0, 'day']
        path_popup = dir_path + '/Sessions/' + day + '/' + str(s) + '/Popup'
        if not os.path.exists(path_popup):
            os.mkdir(path_popup)  
        popup_session.drop(['day', 'session'], axis=1, inplace=True)
        popup_session.to_csv(path_popup + '/popup.csv', index = False)            
               
    
   


def get_date_session_popup(timestamp, path_sessions):
    # Questo metodo calcola la data della sessione in cui è stato effettuato il popup
    # La ricerca viene effettuata nel seguente modo:
    # Si confronta il timestamp del popup con quello della prima sessione della stessa giornata del popup. 
    # Se il popup è stato effettuato prima, allora vuol dire che fa parte dell'ultima sessione della giornata precedente.
    # Altrimenti, il popup è stato effettuato nella sessione con il timestamp più grande minore di quello del popup

    #Datetime del popup
    date_time_popup = datetime.datetime.fromtimestamp(int(timestamp))
    date_popup = datetime.datetime.strftime(date_time_popup, format='%d-%m-%Y')
    
    #Date delle sessioni
    dates = os.listdir(path_sessions)
    dates.sort(key=lambda date: datetime.datetime.strptime(date, "%d-%m-%Y"))
    
    #Giorno prima del popup
    prev_date_time_session = date_time_popup - datetime.timedelta(days=1)
    prev_date_session = datetime.datetime.strftime(prev_date_time_session, format='%d-%m-%Y')
    
    # Potrebbero esserci popup senza aver registrato i dati di E4
    real_date = None
    session = None

    # Se il popup è stato fatto in una sessione
    if date_popup in dates:
        index_date_popup = dates.index(date_popup)
    
        sessions = os.listdir(path_sessions + '/' + date_popup)
        sessions.sort()
        
        # Il popup è stato fatto prima della prima sessione della giornata. In questo caso si ipotizza che
        # sia stato fatto mentre si lavorava ad una sessione a cavallo di due giorni
        if str(timestamp) < sessions[0]:
            #La sessione potrebbe essere l'ultima del giorno precedente
            index_date_popup -= 1
            # Se nel giorno precedente a quello del popup è stata effettuata una sessione
            if dates[index_date_popup] == prev_date_session:
                sessions = os.listdir(path_sessions + '/' + dates[index_date_popup])
                sessions.sort()
                # Controlla se l'ultima sessione è iniziata prima del popup, altrimenti scarta il popup
                if str(timestamp) > sessions[-1]:
                    session = sessions[-1]
                    real_date = dates[index_date_popup]
        else:
            #Cerco la sessione nello stesso giorno del popup
            real_date = dates[index_date_popup]
            for i in range(len(sessions)-1, 0-1, -1):
                if sessions[i] < str(timestamp):
                    session = sessions[i]
                    break
    
    #Questa condizione vale per quelle sessioni fatte a cavallo di due giorni e in cui nel secondo giorno non è stata
    #fatta un'altra sessione
    elif prev_date_session in dates:
        # Se nel giorno precedente a quello del popup è stata effettuata una sessione
        sessions = os.listdir(path_sessions + '/' + prev_date_session)
        sessions.sort()
        # Controlla se l'ultima sessione è iniziata prima del popup, altrimenti scarta il popup
        if str(timestamp) > sessions[-1]:
            session = sessions[-1]
            real_date = prev_date_session


    return real_date, session
    
        
    
    