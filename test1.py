import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from seaborn import *
from datetime import *

def DateandTime(s):
    pattern1='^([0-9][0-9]+)(/)([0-9][0-9]+)(/)([0-9][0-9][0-9][0-9])'
    pattern2='^([0-9]+)(.)([0-9]+)(.)([0-9][0-9][0-9][0-9])'
    result1=re.match(pattern1,s)
    if result1:
        return False
    elif not result1:
        result2=re.match(pattern2,s)
        if result2:
            return True
    else:
        return False

def getDataPoint(line):
    dateTime = line.split(' - ')[0]
    date, time = dateTime.split(', ')
    return date, time

parsedData =[]
chatfilepath = 'WhatsApp.txt'
with open(chatfilepath, encoding="utf-8") as fp:
    fp.readline()
    date, time = None, None
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if DateandTime(line):
            parsedData.append([date, time])
            date, time = getDataPoint(line)
df = pd.DataFrame(parsedData, columns=['Date', 'Time'])
df['m_cnt']=[1]*df.shape[0]
df["Date"] = pd.to_datetime(df["Date"])
df=df.dropna()
df=df.reset_index(drop=True)

df1=df.copy()
days = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thrusday',4:'Friday',5:'Saturday',6:'Sunday'}
df1['Day'] = df1['Date'].dt.weekday.map(days)
df1['Hours'] = df1['Time'].apply(lambda x : x.split(':')[0])

df1 = df1[['Date','Day','Time','Hours','m_cnt']]
df1['Day'] = df1['Day'].astype('category')

hours_lst=list(range(24))
for i in range(24):
    hours_lst[i]=str(hours_lst[i])
for i in range(10):
    hours_lst[i]='0'+hours_lst[i]

a = df1.Hours.unique()
expected = hours_lst
cnt=0
for hour in expected:
    cnt+=1
    if cnt==1:
        if hour not in a:
            df2 = pd.DataFrame({'Date':'2019-03-10','Day':'Sunday','Time':'{}:00'.format(hour),'Hours':hour, 'm_cnt':0}, index=[0])
    else:
        if hour not in a:
            data=pd.DataFrame({'Date':'2019-03-10','Day':'Sunday','Time':'{}:00'.format(hour),'Hours':hour, 'm_cnt':0}, index=[0])
            df2=df2.append(data)





























days_lst = ['Monday','Tuesday','Wednesday','Thrusday','Friday','Saturday','Sunday']
temp_df1=pd.DataFrame(np.zeros((24,5)),columns=['Date','Day','Time','Hours','m_cnt'],dtype=str)
for i in range(24):
    temp_df1['Day'][i]='Monday'
for i in range(24):
    temp_df1['Hours'][i]=hours_lst[i]

temp_df2=pd.DataFrame(np.zeros((24,5)),columns=['Date','Day','Time','Hours','m_cnt'],dtype=str)
for i in range(24):
    temp_df2['Day'][i]='Tuesday'
for i in range(24):
    temp_df2['Hours'][i]=hours_lst[i]

temp_df3=pd.DataFrame(np.zeros((24,5)),columns=['Date','Day','Time','Hours','m_cnt'],dtype=str)
for i in range(24):
    temp_df3['Day'][i]='Wednesday'
for i in range(24):
    temp_df3['Hours'][i]=hours_lst[i]
