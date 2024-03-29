import configparser
import datetime
import os
import zipfile

import numpy as np
import pandas as pd
from bokeh.models import (ColumnDataSource, DatetimeTickFormatter, HoverTool,
                          Span)
from bokeh.plotting import figure

import constant
from signalPreprocess import ACC_HR_Filtering as accelerometer
from signalPreprocess import EDA_Artifact_Detection_Script as eda_artifact
from signalPreprocess import EDA_Peak_Detection_Script as eda_peak


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
        data, fullOutputPath, int(offset), float(thresh), int(start_WT), int(end_WT)
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
    return data_times



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


def get_popup(path_session, date):

    popup_df = popup_process(path_session + '/Popup/' + 'popup.csv')
    popup = extract_popup_date(popup_df, date)


    # print("\n\nRAW TIMESTAMP")
    # print(popup['timestamp'])

    popup['time'] = pd.to_datetime(popup['timestamp'])
    # print("\n\nPARSED TIMESTAMP")
    # print(popup['time'])

    popup["time"] = popup["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    # print("\n\nPARSED TIMESTAMP (localized)")
    # print(popup['time'])

    # popup['time'] = popup['time'].apply(lambda x: x.time())
    return popup


def process_acc(path_session):
    acc, timestamp_0 = accelerometer.load_acc(path_session)
    acc_filter = accelerometer.empatica_filter(acc)
    timestamp = calculate_date_time(timestamp_0,1,len(acc_filter))
    # create a df with the filtered acc data and date and time
    df_acc = pd.DataFrame(acc_filter, columns=['acc_filter'])
    df_acc['timestamp'] = timestamp
    df_acc['timestamp'] = pd.to_datetime(df_acc['timestamp'])
    return df_acc

def process_hr(path_session):
    hr, timestamp_0_hr = accelerometer.load_hr(path_session)
    timestamp = calculate_date_time(timestamp_0_hr, 1, len(hr))
    df_hr = pd.DataFrame(hr, columns=['hr'])
    df_hr['timestamp'] = timestamp
    df_hr['timestamp'] = pd.to_datetime(df_hr['timestamp'])

    return df_hr

def get_session_EDA_ACC_TEMP(path_session):
    EDA_df = pd.read_csv(path_session + '/Data/' + 'EDA.csv')
    ACC_df = pd.read_csv(path_session + '/Data/' + 'ACC.csv')
    TEMP_df = pd.read_csv(path_session + '/Data/' + 'TEMP.csv')


    return EDA_df, ACC_df, TEMP_df


def save_EDAs_filtered(path_days, thresh, offset, start_WT, end_WT):
    days = os.listdir(path_days)
    for d in days:
        path_sessions = path_days + '/' + d + '/'
        sessions = os.listdir(path_sessions)
        for s in sessions:
            path_session = path_days + '/' + d  + '/' + s
            EDA, ACC, TEMP = get_session_EDA_ACC_TEMP(path_session)
            artifact_file = os.path.join(constant.artifact_output_path, "artifact_detected.csv")
            output_file_path = os.path.join(constant.artifact_output_path, "result.csv")
            classify_artifacts(EDA, ACC, TEMP, artifact_file, output_file_path)

            data = detect_peak(output_file_path, artifact_file, thresh, offset, start_WT, end_WT)
            data.reset_index(inplace=True, drop=True)
            data['timestamp'] = pd.to_datetime(data['timestamp'])\
                                  .dt.tz_localize('UTC')\
                                  .dt.tz_convert('Europe/Berlin')


            data.to_csv(path_session + '/Data/data_eda_filtered.csv', index=False)






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


def read_param_EDA():
    config_data = configparser.ConfigParser()
    config_data.read("config.ini")

    params = config_data['PARAMETERS PEAKS EDA']
    offset = int(params['OFFSET'])
    thresh = float(params['THRESH'])
    start_WT = int(params['START_WT'])
    end_WT = int(params['END_WT'])

    return offset, thresh, start_WT, end_WT


def create_fig_line(df_sign, x, y, title, y_axis_label, sign, df_popup):
    fig_sign = figure(plot_height=400, x_axis_type='datetime',
                    title=title, x_axis_label='Time', y_axis_label=y_axis_label,
                    sizing_mode='stretch_both', tools = ['pan', 'xpan', 'box_zoom' ,'reset', 'save'])

    #Rimozione della griglia dal background
    fig_sign.xgrid.grid_line_color = None
    fig_sign.ygrid.grid_line_color = None
    data_src_sign = ColumnDataSource(df_sign)
    line_plot_sign = fig_sign.line(x=x, y=y, source=data_src_sign)


    line_hover = HoverTool(renderers=[line_plot_sign],
                            tooltips=[(sign, "@"+y), ("Time", "@time{%H:%M:%S}")],
                            formatters={'@time': 'datetime'})
    fig_sign.add_tools(line_hover)



    #Mean
    mean = df_sign.loc[:, y].mean()
    fig_sign.add_layout(Span(location=mean, dimension='width', line_color="red", line_alpha=0.5, line_width=1, line_dash='dashed'))


    if df_popup is not None:
        # Il seguente controllo è necessario per assegnare ai popup dei timestamp esistenti nei segnali (per la visualizzazione)
        # Assegno i valori dei segnali ai popup nei relativi timestamp
        df_temp = df_sign.copy()
        df_popup_copy = df_popup.copy()


        #Assegnazione dei valori y ai popup
        df_popup_copy[y] = None
        df_popup_copy["time"] = df_popup_copy["time"].dt.tz_localize(None)
        for i in range(df_popup_copy.shape[0]):
            time = df_popup_copy.loc[i,'time']
            temp = df_temp[df_temp[x] == time]
            if not temp.empty:
                temp.reset_index(inplace=True, drop=True)

                # Assegno il valore del segnale
                df_popup_copy.loc[i, y] = temp.loc[0,y]
                # Assegno il valore del timestamp
                # df_popup_copy.loc[i, x] = temp.loc[0,x]

        #Se ci sono popup con time che non sono presenti nei segnali, non vengono considerati
        df_popup_copy = df_popup_copy[df_popup_copy[y].notna()]


        #Sostituzione dei valori nulli nelle note con stringa vuota. Necessario per la visualizzazione sull'HoverTool
        df_popup_copy['notes'] = df_popup_copy['notes'].astype(str)
        df_popup_copy.loc[df_popup_copy["notes"] == 'nan', 'notes'] = ''


        datasrc = ColumnDataSource(df_popup_copy)
        circle_plot = fig_sign.circle(name='report', x=x, y=y, source=datasrc, fill_color="yellow",
                                size=9)
        circle_hover = HoverTool(renderers=[circle_plot],
                                tooltips=[("Activity", "@activity"), ("Valence", "@valence"), ("Arousal", "@arousal"),
                                            ("Dominance", "@dominance"), ("Productivity", "@productivity"),
                                            ("Note", "@notes"), ("Time", "@time{%H:%M:%S}"), (sign, "@"+y)],
                                    formatters={'@time': 'datetime'})
        fig_sign.add_tools(circle_hover)

    # Configurazione dei valori visualizzati sotto l'ascissa nel formato HH:MM
    fig_sign.xaxis.formatter = DatetimeTickFormatter(
        hourmin = ['%H:%M'],
        hours = ['%H:%M'],
        days = ['%H:%M'],
        months = ['%H:%M'],
        years = ['%H:%M'],
        minutes = ['%H:%M'],
        seconds = ['%H:%M'],
        minsec = ['%H:%M'],
        microseconds = ['%H:%M'],
        milliseconds = ['%H:%M']
    )
    return fig_sign

def extract_popup_date(popup, date):
    #Rimuove i popup relativi agli altri giorni
    for i in range(popup.shape[0]):

        date_temp = popup.loc[i, 'timestamp'].date()
        date_temp = datetime.datetime.strftime(date_temp, "%d-%m-%Y")

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
    dir_popup = dir_path  + '/Popup'
    if len(os.listdir(dir_popup)) > 0:
        all_popup = pd.DataFrame(columns = ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity',
                                   'status_popup', 'notes'])
        for file in os.listdir(dir_popup):
            popup = pd.read_csv(dir_popup + '/' + file, header=None, encoding= 'unicode_escape')
            popup.columns = ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity',
                                   'status_popup', 'notes']
            all_popup = pd.concat([all_popup, popup])

        all_popup = all_popup[all_popup['status_popup'] == 'POPUP_CLOSED']
        all_popup.drop_duplicates(inplace=True)
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



