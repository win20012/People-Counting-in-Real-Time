import pandas as pd
import datetime
import argparse
#from mylib.config import cam_place

def data_converter(enter,exit,excel_name):
    df=pd.read_excel(excel_name)
    ex_time=[f'{i}:00:00' for i in range(24)]
    dt_obj=[datetime.datetime.strptime(str(i),'%H:%M:%S') for i in df.iloc[:,0]]
    for i in dt_obj:
        if i.time() >= datetime.datetime.now().time() :
            j=dt_obj.index(i)
            rownum=j-1
            break
        elif datetime.datetime.now().time() >= dt_obj[-1].time():
            rownum=int(23)
            break
        else:
            continue
    #print(df.T)
    #print(df.columns[-1])
    #still roday
    if datetime.datetime.now().strftime('%Y-%m-%d') != df.columns[-1]:
        data=['NA' if i != rownum else enter-exit for i in range(24)]
        df[str(datetime.datetime.now().strftime('%Y-%m-%d'))]=data
        df.index=ex_time
        df.index.name = 'Time'
        df.drop('Time',axis=1,inplace=True)

    #new day
    else:
        today_data=df.loc[:,str(datetime.datetime.now().strftime('%Y-%m-%d'))]
        today_data=today_data.to_list()
        try:
            today_data[rownum]=enter-exit
        except UnboundLocalError:
            today_data[0]=enter-exit

        df[datetime.datetime.now().strftime('%Y-%m-%d')]=today_data
        df.index=ex_time
        df.index.name = 'Time'
        df.drop('Time',axis=1,inplace=True)
        #print(df)
        #print(today_data)
    #df2=pd.DataFrame()
    df.to_excel(excel_name)

#data_converter(15,2)

def create_summary(enter,exit,excel_name):
    #data={'櫃位地點':cam_place,'People Enter':info[1][1],'People Exit':info[0][1],'Current People Inside':info2[0][1],'Date':datetime.datetime.now()}
    
    ex_time=[f'{i}:00:00' for i in range(24)]
    #print(ex_time)
    dt_obj=[datetime.datetime.strptime(str(i),'%H:%M:%S') for i in ex_time]
    for i in dt_obj:
        if i.time() >= datetime.datetime.now().time() :
            j=dt_obj.index(i)
            rownum=j-1
            break
        elif datetime.datetime.now().time() >= dt_obj[-1].time():
            rownum=int(23)
            break
        else:
            continue
    #print(ex_time)    
    #print(datetime.datetime.now())
    #print(rownum)
    datal3=['NA' if i != rownum else enter-exit for i in range(24)]
    #print(datal)
    data={datetime.datetime.now().strftime('%Y-%m-%d'):datal3}
    #print(data)
    df=pd.DataFrame(data=data,index=ex_time,columns=[datetime.datetime.now().strftime('%Y-%m-%d')])
    df.index.name = 'Time'
    df.to_excel(excel_name)
#create_summary(5,2)


