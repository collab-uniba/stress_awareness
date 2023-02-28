import subprocess
import os
import datetime

import numpy as np
import pandas as pd
import sys

import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './eda_explorer')
import importlib
eda_artifact = importlib.import_module("EDA-Artifact-Detection-Script")
eda_peak = importlib.import_module("EDA-Artifact-Detection-Script")

'''
 Please note that this script use scripts released by Taylor et al. that you can find here: https://github.com/MITMediaLabAffectiveComputing/eda-explorer
 
  Taylor, Sara et al. “Automatic identification of artifacts in electrodermal activity data.” 
  Annual International Conference of the IEEE Engineering in Medicine and Biology Society. 
  IEEE Engineering in Medicine and Biology Society. 
  Annual International Conference vol. 2015 (2015): 1934-7. 
  doi:10.1109/EMBC.2015.7318762

'''


def calc_y(a,c ):
    b = []
    for _ in range(0, 6):
        y = (a * c / 100)
        b.append(-y)
        c += 10
    return b

mydir = os.getcwd()
mydir_tmp = mydir + "//eda_explorer"
mydir_new = os.chdir(mydir_tmp)
mydir = os.getcwd()


input_path =r"..//LabStudy/S1/Data/2giugno2022_150205_A02886"
artifact_output_path = r"../LabStudy/S1/Data/2giugno2022_150205_A02886"
popup_path = r"../LabStudy/S1/popup/popup/2022-05-31.csv"

artifact_path = os.path.join(artifact_output_path, "artifact_detected.csv")
ouput_path = os.path.join(artifact_output_path, "result.csv")

numClassifiers = 1
classifierList = ['Binary']
labels, data = eda_artifact.classify(classifierList, input_path)
fullOutputPath = artifact_path

if fullOutputPath[-4:] != '.csv':
    fullOutputPath = fullOutputPath + '.csv'

featureLabels = pd.DataFrame(labels, index=pd.date_range(start=data.index[0], periods=len(labels), freq='5s'),
                                 columns=classifierList)

featureLabels.reset_index(inplace=True)
featureLabels.rename(columns={'index': 'StartTime'}, inplace=True)
featureLabels['EndTime'] = featureLabels['StartTime'] + datetime.timedelta(seconds=5)
featureLabels.index.name = 'EpochNum'

cols = ['StartTime', 'EndTime']
cols.extend(classifierList)

featureLabels = featureLabels[cols]
featureLabels.rename(columns={'Binary': 'BinaryLabels', 'Multiclass': 'MulticlassLabels'},
                         inplace=True)

featureLabels.to_csv(fullOutputPath)
data.to_csv(ouput_path)

#subprocess.call(" ".join(["python EDA-Artifact-Detection-Script.py", input_path, artifact_path, ouput_path]), shell=True)

# open csv with raw signal and filtered signal
signal_df = pd.read_csv(ouput_path,  names = ['timestamp', 'EDA', 'filtered_eda', 'AccelX', 'AccelY', 'AccelZ', 'Temp'])

artifact_df = pd.read_csv(artifact_path)
signal_df['timestamp'] = signal_df['timestamp'].astype('datetime64[ns]')
artifact_df['StartTime'] = artifact_df['StartTime'].astype('datetime64[ns]')
# merge output of Artifact script with df with eda signal
eda_clean = pd.merge(signal_df, artifact_df, how = 'outer', left_on='timestamp', right_on='StartTime')
# forward filling of na values due to the merge
eda_clean = eda_clean.fillna(method = 'ffill')

x = eda_clean['filtered_eda'].values
dx = eda_clean['BinaryLabels']
# set nan if the bianary labels is -1 else set the values that is already there
filt = [np.nan if t == -1.0 else y for y,t in zip(x,dx)]

eda_clean['filtered_eda'] = filt

# substitute the nan value with the previous trustable
eda_clean['filtered_eda'] = eda_clean['filtered_eda'].ffill()

# if the very first value of df is labeled with -1 it is now set with 'filtered_eda';
# we delete this row;
# in other words we delete the row values that have no previous trustable values.

eda_clean = eda_clean[~eda_clean['filtered_eda'].isin(['filtered_eda'])]
final_df = eda_clean[['timestamp', 'filtered_eda']]

final_df.to_csv(artifact_output_path + '/filtered_eda.csv', index = False)


subprocess.call("python EDA-Peak-Detection-Script.py "
                  + artifact_output_path + "/filtered_eda.csv "
                  + artifact_output_path + "/result_peak.csv "
                  + popup_path, shell=True)


