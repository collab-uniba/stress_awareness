from load_files import loadData_E4
import pandas as pd
import numpy as np
import os
import re
import scipy.signal as scisig
import matplotlib.pyplot as plt
import pprint

fullOutputPath = r'C:\Users\user\Desktop\FieldStudy\ASML\P18\week1\10_31_2019, 12_00_38\find_peak.csv'
filepath = r'C:\Users\user\Desktop\FieldStudy\ASML\P18\week1\10_31_2019, 12_00_38'
filepath_confirm = os.path.join(filepath, "EDA.csv")
data = loadData_E4(filepath)
data = data.drop(['EDA', 'Temp', 'AccelX', 'AccelZ', 'AccelY'], axis=1)
artifact = pd.read_csv (r'C:\Users\user\Desktop\FieldStudy\ASML\P18\week1\10_31_2019, 12_00_38\artifact_detection.csv')

artifact['StartTime'] = artifact['StartTime'].astype('datetime64[ns]')
data['Timestamp'] = data.index
data['Timestamp'] = data['Timestamp'].astype('datetime64[ns]')

result_data = (pd.merge(data, artifact, left_on='Timestamp', right_on='StartTime', how='outer'))
result_data = result_data.drop(['StartTime', 'EpochNum', 'EndTime'], axis=1)
result_data = result_data.ffill()
result_data.to_csv(r'C:\Users\user\Desktop\FieldStudy\ASML\P18\week1\10_31_2019, 12_00_38\artifact_detection_comparison.csv', sep=',')

print(result_data)
print(data)
