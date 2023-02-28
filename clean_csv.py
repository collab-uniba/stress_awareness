import os
import pandas as pd
import datetime

directory = r'path/to/self-report/files'

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


subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
for files in subfolders:
    for f in os.scandir(files):
        if f.is_dir():
            if os.path.exists(os.path.join(f, "merged.csv")):
                os.remove(os.path.join(f, 'merged.csv'))
                print(os.path.join(f, 'merged.csv'))
            all_files = []
            for popup in os.listdir(f):
                if popup.endswith('.csv'):
                    path_participant = os.path.join(f, popup)
                    all_files.append(os.path.join(f, popup))
                    print(all_files)
                    df_merged = pd.concat([pd.read_csv(f, sep=',', names = ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity', 'status_popup','notes']) for f in all_files])
                    df_merged = df_merged.drop_duplicates()
                    print(df_merged)
                    first_column = df_merged['timestamp']
                    df_merged['timestamp'] = get_datetime_filename(first_column)
                    df_merged = df_merged.sort_values(by="timestamp")
                    #print(df_merged['timestamp'])
                    df_merged.to_csv(os.path.join(f, 'merged.csv'),  index=None, header = ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity', 'status_popup','notes'], sep = ';')
