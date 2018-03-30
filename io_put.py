# 头文件
import numpy as np
import pandas as pd
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
#data=np.loadtxt(open('TotalFeatures-ISCXFlowMeter.csv','rb'),delimiter=',',skiprows=0)
data = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv')
data=data.values

Begin=np.arange(82)
GM=np.arange(82)
Adware=np.arange(82)

for i in range(data.shape[0]):
    if data[i,81]==0:
        Begin=np.row_stack((Begin,data[i,:]))
    if data[i,81]==1:
        Adware=np.row_stack((Adware,data[i,:]))
    if data[i,81]==2:
        GM=np.row_stack((GM,data[i,:]))

np.savetxt('Begin.csv',Begin,delimiter=',')
np.savetxt('Adware.csv',Adware,delimiter=',')
np.savetxt('GM.csv',GM,delimiter=',')

pd_data = pd.DataFrame(Begin,
                       columns=['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin',
                                'bVarianceDataBytes',
                                'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward', 'calss'])
pd_data.to_csv('Begin.csv')