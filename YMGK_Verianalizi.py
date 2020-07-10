# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:52:26 2020

@author: RamazanSamet
"""

import pandas as pd
import datetime
import numpy as np
import joblib
import math
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

airQuality= pd.read_excel("C:/Users/samet yöntürk/Desktop/RamazanSamet.xlsx")


def Average(lst):
    return sum(lst) / len(lst)

def check_if_weekend(today):
    try:
        isinstance(today, datetime.datetime)
        upper_limit = today + datetime.timedelta(days=(6 - today.weekday()))
        lower_limit = today + datetime.timedelta(days=(5 - today.weekday()))
        if today >= lower_limit <= upper_limit:
            return True
        else:
            return False
    except ValueError:
        print('Your date is not a datetime object.')



def calAQISO2(so2):
    soi2 = 0
    if (so2 >= 0 and so2 <= 100):
        soi2 = ((50 - 0) / (100 - 0)) * (so2 - 0) + 0
    if (so2 >= 101 and so2 <= 250):
        soi2 = ((100 - 51) / (250 - 101)) * (so2 - 101) + 51
    if (so2 >= 251 and so2 <= 500):
        soi2 = ((150 - 101) / (500 - 251)) * (so2 - 251) + 101
    if (so2 >= 501 and so2 <= 850):
        soi2 = ((200 - 151) / (850 - 501)) * (so2 - 501) + 151
    if (so2 >= 851 and so2 <= 1100):
        soi2 = ((300 - 201) / (1100 - 851)) * (so2 - 851) + 201
    if (so2 >= 1101 and so2 <= 1500):
        soi2 = ((500 - 301) / (1500 - 1101)) * (so2 - 1101) + 301

    return soi2


def calAQINO2(no2):
    noi2 = 0
    if (no2 >= 0 and no2 <= 100):
        noi2 = ((50 - 0) / (100 - 0)) * (no2 - 0) + 0
    if (no2 >= 101 and no2 <= 200):
        noi2 = ((100 - 51) / (200 - 101)) * (no2 - 101) + 51
    if (no2 >= 201 and no2 <= 500):
        noi2 = ((150 - 101) / (500 - 201)) * (no2 - 201) + 101
    if (no2 >= 501 and no2 <= 1000):
        noi2 = ((200 - 151) / (1000 - 501)) * (no2 - 501) + 151
    if (no2 >= 1001 and no2 <= 2000):
        noi2 = ((300 - 201) / (2000 - 1001)) * (no2 - 1001) + 201
    if (no2 >= 2001 and no2 <= 3000):
        noi2 = ((500 - 301) / (3000 - 2001)) * (no2 - 2001) + 301
    return noi2


def calAQIPM10(pm10):
    pm10i2 = 0
    if (pm10 >= 0 and pm10 <= 50):
        pm10i2 = ((50 - 0) / (50 - 0)) * (pm10 - 0) + 0
    if (pm10 >= 51 and pm10 <= 100):
        pm10i2 = ((100 - 51) / (100 - 51)) * (pm10 - 51) + 51
    if (pm10 >= 101 and pm10 <= 260):
        pm10i2 = ((150 - 101) / (260 - 101)) * (pm10 - 101) + 101
    if (pm10 >= 261 and pm10 <= 400):
        pm10i2 = ((200 - 151) / (400 - 261)) * (pm10 - 261) + 151
    if (pm10 >= 401 and pm10 <= 520):
        pm10i2 = ((300 - 201) / (520 - 401)) * (pm10 - 401) + 201
    if (pm10 >= 521 and pm10 <= 620):
        pm10i2 = ((500 - 301) / (620 - 521)) * (pm10 - 521) + 301

    return pm10i2



def calAQIPM25(pm25):
    pm25i2 = 0
    if (pm25 >= 0 and pm25 <= 12):
        pm25i2 = ((50 - 0) / (12 - 0)) * (pm25 - 0) + 0
    if (pm25 >= 12.1 and pm25 <= 35.4):
        pm25i2 = ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    if (pm25 >= 35.5 and pm25 <= 55.4):
        pm25i2 = ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    if (pm25 >= 55.5 and pm25 <= 150.4):
        pm25i2 = ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    if (pm25 >= 150.5 and pm25 <= 250.4):
        pm25i2 = ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
    if (pm25 >= 250.5 and pm25 <= 350.4):
        pm25i2 = ((400 - 301) / (350.4 - 250.5)) * (pm25 - 250.5) + 301
    if (pm25 >= 350.5 and pm25 <= 505.4):
        pm25i2 = ((500 - 401) / (505.4 - 350.5)) * (pm25 - 350.5) + 401
    return pm25i2



def calAQICO(CO):
    coi2 = 0
    if (CO >= 0 and CO <= 5500):
        coi2 = ((50 - 0) / (5500 - 0)) * (CO - 0) + 0
    if (CO >= 5501 and CO <= 10000):
        coi2 = ((100 - 51) / (10000 - 5501)) * (CO - 5501) + 51
    if (CO >= 10001 and CO <= 16000):
        coi2 = ((150 - 101) / (16000 - 10001)) * (CO - 10001) + 101
    if (CO >= 16001 and CO <= 24000):
        coi2 = ((200 - 151) / (24000 - 16001)) * (CO - 16001) + 151
    if (CO >= 24001 and CO <= 32000):
        coi2 = ((300 - 201) / (32000 - 24001)) * (CO - 24001) + 201
    if (CO >= 32001 and CO <= 40000):
        coi2 = ((500 - 301) / (40000 - 32001)) * (CO - 32001) + 301
    print (coi2)
    return coi2



def calAQIO3(O3):
    o3i2 = 0
    if (O3 >= 0 and O3 <= 120):
        o3i2 = ((50 - 0) / (120 - 0)) * (O3 - 0) + 0
    if (O3 >= 121 and O3 <= 160):
        o3i2 = ((100 - 51) / (160 - 121)) * (O3 - 121) + 51
    if (O3 >= 161 and O3 <= 180):
        o3i2 = ((150 - 101) / (180 - 161)) * (O3 - 161) + 101
    if (O3 >= 181 and O3 <= 240):
        o3i2 = ((200 - 151) / (240 - 181)) * (O3 - 181) + 151
    if (O3 >= 241 and O3 <= 700):
        o3i2 = ((300 - 201) / (700 - 241)) * (O3 - 241) + 201
    if (O3 >= 701 and O3 <= 1700):
        o3i2 = ((500 - 301) / (1700 - 701)) * (O3 - 701) + 301
    return o3i2






airQuality["Day"] = [da.day for da in airQuality["Tarih"]]
airQuality["Month"] = [da.month for da in airQuality["Tarih"]]
airQuality["Year"] = [da.year for da in airQuality["Tarih"]]
airQuality["Hour"] = [da.hour for da in airQuality["Tarih"]]
airQuality["Minute"] = [da.minute for da in airQuality["Tarih"]]
airQuality["Second"] = [da.second for da in airQuality["Tarih"]]
airQuality.fillna(0,inplace=True)

rowCount = len(airQuality.index)
hoursPerDay = 24
dayCount = int(rowCount / hoursPerDay)

dictWeekDayAverage = dict()
dictWeekEndAverage = dict()


for i in airQuality:
    if(i != 'Tarih'):
        print(i,'min', min((float(sub) for sub in  airQuality[i])))
        print(i,'max', max((float(sub) for sub in  airQuality[i])))



def DispatchAverages():
    for i in range(1, 30):
        airQualityDay = airQuality.loc[airQuality["Day"] == i]
        dayRowCount = len(airQualityDay.index)

        timestamp = airQualityDay["Tarih"].iloc[0]
        datetime_object = timestamp.to_pydatetime()
        isWeekend = check_if_weekend(datetime_object)

        dictAverage = None
        if(isWeekend):
            dictAverage = dictWeekEndAverage
        else:
            dictAverage = dictWeekDayAverage

        averagePM10 = airQualityDay["PM10"].sum() / dayRowCount
        averageSO2 = airQualityDay["SO2"].sum() / dayRowCount
        averageCO = airQualityDay["CO"].sum() / dayRowCount
        averageNO2 = airQualityDay["NO2"].sum() / dayRowCount
        averageNOX = airQualityDay["NOX"].sum() / dayRowCount
        AQIDay = CalculateAQI(averagePM10,averageSO2,averageCO,averageNO2)

        def insertToDict(key, val):
            if key in dictAverage:
                l = dictAverage[key]
                l.append(val)
            else:
                dictAverage[key] = list()
                dictAverage[key].append(val)

        insertToDict("PM10", averagePM10)
        insertToDict("SO2", averageSO2)
        insertToDict("CO", averageCO)
        insertToDict("NO2", averageNO2)
        insertToDict("NOX", averageNOX)
        insertToDict("AQI", AQIDay)


def CalculateAQI(pm10, so2, co, no2 ):
    aqiPm10 = calAQIPM10(pm10)
    aqiSo2 = calAQISO2(so2)
    aqiCo = calAQICO(co)
    aqiNo2 = calAQINO2(no2)

    aqiList = [aqiPm10, aqiSo2, aqiCo, aqiNo2 ]
    return max(aqiList)


def AverageDifference():
    weekDayPM10 = Average(dictWeekDayAverage["PM10"])
    weekEndPM10 = Average(dictWeekEndAverage["PM10"])
    print("PM10 Hafta Sonu : {0} , PM10 Hafta Içi : {1}".format(str(weekDayPM10), str(weekEndPM10)))

    weekDaySO2 = Average(dictWeekDayAverage["SO2"])
    weekEndSO2 = Average(dictWeekEndAverage["SO2"])
    print("SO2 Hafta Sonu : {0} , SO2 Hafta Içi : {1}".format(str(weekDaySO2), str(weekEndSO2)))

    weekDayNO2 = Average(dictWeekDayAverage["NO2"])
    weekEndNO2 = Average(dictWeekEndAverage["NO2"])
    print("NO2 Hafta Sonu : {0} , NO2 Hafta Içi : {1}".format(str(weekDayNO2), str(weekEndNO2)))

    weekDayNOX = Average(dictWeekDayAverage["NOX"])
    weekEndNOX = Average(dictWeekEndAverage["NOX"])
    print("NOX Hafta Sonu : {0} , NOX Hafta Içi : {1}".format(str(weekDayNOX), str(weekEndNOX)))


    weekDayAQI = Average(dictWeekDayAverage["AQI"])
    weekEndAQI = Average(dictWeekEndAverage["AQI"])
    print("AQI Hafta Sonu : {0} , AQI Hafta Içi : {1}".format(str(weekDayAQI), str(weekEndAQI)))



DispatchAverages()
AverageDifference()


def create_dataset(filename):
    curr_day = 1
    start_index = 0
    curr_index = curr_day * hoursPerDay

    file = open('{0}.csv'.format(filename), 'w')
    writer = csv.writer(file)
    writer.writerow(["AQI", "Class"])

    while(curr_day <= dayCount):
        airQualityDay = airQuality.iloc[start_index:curr_index]
        dayRowCount = len(airQualityDay.index)
        ##

        timestamp = airQualityDay["Tarih"].iloc[0]
        datetime_object = timestamp.to_pydatetime()
        isWeekend = check_if_weekend(datetime_object)

        averagePM10 = airQualityDay["PM10"].sum() / dayRowCount
        averageSO2 = airQualityDay["SO2"].sum() / dayRowCount
        averageCO = airQualityDay["CO"].sum() / dayRowCount
        averageNO2 = airQualityDay["NO2"].sum() / dayRowCount
        averageNOX = airQualityDay["NOX"].sum() / dayRowCount


        Date = "{0}.{1}.{2}".format(timestamp.day,timestamp.month,timestamp.year)
        AQI = CalculateAQI(averagePM10,averageSO2,averageCO,averageNO2)
        Class = 1 if isWeekend else 0

        writer.writerow([AQI, Class])

        ##
        curr_day += 1
        start_index = curr_index
        curr_index = curr_day * hoursPerDay

    file.close()

create_dataset("model_dataset")
