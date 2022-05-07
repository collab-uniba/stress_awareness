import subprocess
import sys
import os
import numpy as np
from datetime import datetime

import pandas as pd
mydir = os.getcwd() # would be the MAIN folder
mydir_tmp = mydir + "//eda_explorer" # add the testA folder name
mydir_new = os.chdir(mydir_tmp) # change the current working directory
mydir = os.getcwd() # set the main directory again, now it calls testA

input_path ="C:/Users/user/Desktop/FieldStudy/GREEFA/P13/week2/10_18_2019,08_16_50"
artifact_output_path = "C:/Users/user/Desktop/FieldStudy/GREEFA/P13/week2/10_18_2019,08_16_50"
popup_path = "C:/Users/user/Desktop/FieldStudy/GREEFA/P13/popup/2019-10-18.csv"
# Calling Artifact Detection Script
# subprocess.call("python EDA-Artifact-Detection-Script.py "
#                 + input_path + " " + artifact_output_path + "artifact_detected.csv "
#                 + artifact_output_path + "result.csv", shell=True)

# open artifact csv result
artifact_df = pd.read_csv(artifact_output_path + "/artifact_detected.csv")

# open csv with raw signal and filtered signal
signal_df = pd.read_csv(artifact_output_path + "/result.csv",  names = ['timestamp', 'EDA', 'filtered_eda', 'AccelX', 'AccelY', 'AccelZ', 'Temp'])

print(signal_df)

signal_df['timestamp'] = signal_df['timestamp'].astype('datetime64[ns]')
artifact_df['StartTime'] = artifact_df['StartTime'].astype('datetime64[ns]')

# merge output of Artifact script with df with eda signal
eda_clean = pd.merge(signal_df, artifact_df, how = 'outer', left_on='timestamp', right_on='StartTime')
print(eda_clean)
# forward filling of na values due to the merge
eda_clean = eda_clean.fillna(method = 'ffill')




x = eda_clean['filtered_eda'].values
dx = eda_clean['BinaryLabels']
# set nan if the bianary labels is -1 else set the values that is already there
filt = [np.nan if t == -1.0 else y for y,t in zip(x,dx)]

eda_clean['filtered_eda'] = filt

# substitute the nan value with the previous trustable
eda_clean['filtered_eda'] = eda_clean['filtered_eda'].ffill()

# if the very first value of df was labeled with -1 it is now set with 'filtered_eda'; I delete this row;
# in other words I delete the row values that have no previous trustable values.

eda_clean = eda_clean[~eda_clean['filtered_eda'].isin(['filtered_eda'])]
#eda_clean['timestamp'] = pd.to_datetime(eda_clean['timestamp'].values, utc=True).tz_convert('Europe/Berlin')
#eda_clean['timestamp'] = eda_clean['timestamp'].dt.strftime(fmt)
final_df = eda_clean[['timestamp', 'filtered_eda']]
print('final_df', final_df)
final_df.to_csv(artifact_output_path + '/eda_clean.csv', index = False)

#Calling peak Detection Script
subprocess.call("python EDA-Peak-Detection-Script.py "
                  + artifact_output_path + "/eda_clean.csv "
                  + artifact_output_path + "/result_peak.csv "
                  + popup_path,shell=True)