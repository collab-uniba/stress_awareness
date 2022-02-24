import os
import pandas as pd
import datetime
import shutil

directory = r'C:\Users\user\Desktop\FieldStudy'


def get_datetime_filename(column):
    human_timestamp = []
    for value in column:
        human_date = datetime.datetime.fromtimestamp(int(value))
        human_timestamp.append(human_date)
    return human_timestamp


d = []
subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
for files in subfolders:
    # print(files)
    for f in os.scandir(files):
        if f.is_dir():
            all_files = []
            for popup in os.listdir(f):
                if os.path.exists(os.path.join(f, 'merged.csv')):
                    popup_df = pd.read_csv(os.path.join(f, 'merged.csv'), sep=';')
                    popup_df['timestamp'] = pd.to_datetime(popup_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
                    for idx, row in popup_df.iterrows():
                        if idx + 1 < len(popup_df['timestamp']):
                            #print(row['timestamp'].date())
                            d.append(
                                {
                                    'timestamp': row['timestamp'],
                                    'activity': row['activity'],
                                    'valence': row['valence'],
                                    'arousal': row['arousal'],
                                    'dominance': row['dominance'],
                                    'productivity': row['productivity'],
                                    'status_popup': row['status_popup'],
                                    'notes': row['notes']
                                }
                            )
                            if row['timestamp'].date() < popup_df['timestamp'][idx + 1].date():
                                print(row['timestamp'].date(), popup_df['timestamp'][idx + 1].date())

                                final_filename = ('popup//' + str(row['timestamp'].date()) + '.csv')
                                df = pd.DataFrame(d, columns=['timestamp', 'activity', 'valence', 'arousal', 'dominance',
                                                               'productivity', 'status_popup', 'notes'])
                                print(df)

                                if not os.path.exists(os.path.join(f, 'popup')):
                                    #.rmtree(os.path.join(f, 'popup'))
                                    os.makedirs(os.path.join(f, 'popup'))
                                final_filename = (str(row['timestamp'].date()) + '.csv')
                                #print(final_filename)
                                df.to_csv(os.path.join(*[f,'popup',final_filename]),  index=None, header = ['timestamp', 'activity', 'valence', 'arousal', 'dominance', 'productivity', 'status_popup','notes'], sep = ';')

                                d = []
