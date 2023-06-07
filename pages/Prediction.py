import numpy as np
import streamlit as st
import pickle
import pandas as pd
import json
import time
import sys
from os import system
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as ox



model_file_name = r"C:\Users\YR272YB\OneDrive - EY\Desktop\Projects\Utility_AHI\model_store\xgb_classification_v2.pkl"
xgb_model_loaded = pickle.load(open(model_file_name, "rb"))
print("Model Loaded Successfully.")

try:
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
    a = pd.read_csv(uploaded_files)
    b = a.copy()
    st.write(b)
    m = ['Feeder_Category',
         'CABLE_SIZE',
         'No_of_times_exceeded_60_',
         'LENGTH_OF_CABLE_MTRS_',
         'NO__OF_JOINTS',
         'AGE_YRS_',
         'PILC',
         'XLPE',
         'Average_Current',
         'Peak_Current',
         'Average_voltage',
         'Peak_Voltage',
         'Avg_Heat_Index',
         'Peak_HI',
         'Avg_humidity',
         'Max_Humidity',
         'Avg_TEMP',
         'Max_TEMP',
         'New_Summer',
         'New_Monsoon',
         'New_Winter']
    if st.checkbox("Before going for prediction please ensure you have required number of features."):
        for i in m:
            if i not in a.columns:
                st.write(f"Feature {i} is missing ")
            else:
                pass
        st.write("Dataset has all the required features")
        if st.button("Predict"):
            d = b[m]
            d['Feeder_Category'].fillna('RES', inplace=True)
            d['NO__OF_JOINTS'].fillna(d['NO__OF_JOINTS'].mean(), inplace=True)
            d['AGE_YRS_'].fillna(d['AGE_YRS_'].mean(), inplace=True)

            d['Feeder_Category'] = d['Feeder_Category'].map({
                'RES': 0,
                'COM': 1,
                'MIX': 2})
            d['New_Summer'] = d['New_Summer'].map({
                'N': 0,
                'Y': 1})
            d['New_Monsoon'] = d['New_Monsoon'].map({
                'N': 0,
                'Y': 1})
            d['New_Winter'] = d['New_Winter'].map({
                'N': 0,
                'Y': 1})
            ## CABLE_SIZE
            d['CABLE_SIZE'] = d['CABLE_SIZE'].map({
                240.00: 0,
                300.00: 1,
                225.00: 1,
                120.00: 2,
                185.00: 2,
                70.00: 2,
                0.15: 2,
                0.20: 2,
                0.30: 2})

            a = ['No_of_times_exceeded_60_', 'LENGTH_OF_CABLE_MTRS_', 'NO__OF_JOINTS', 'AGE_YRS_',
                 'Average_Current', 'Peak_Current',
                 'Average_voltage', 'Peak_Voltage',
                 'Avg_Heat_Index', 'Peak_HI',
                 'Avg_humidity', 'Max_Humidity',
                 'Avg_TEMP', 'Max_TEMP']

            # compute interquantile range to calculate the boundaries
            lower_boundries = []
            upper_boundries = []
            for i in a:
                IQR = d[i].quantile(0.75) - d[i].quantile(0.25)
                lower_bound = d[i].quantile(0.25) - (1.5 * IQR)
                upper_bound = d[i].quantile(0.75) + (1.5 * IQR)
                print(i, ":", lower_bound, ",", upper_bound)
                lower_boundries.append(lower_bound)
                upper_boundries.append(upper_bound)

            # replace the all the outliers which is greater then upper boundary by upper boundary
            j = 0
            for i in a:
                d.loc[d[i] > upper_boundries[j], i] = int(upper_boundries[j])
                j = j + 1

            p = xgb_model_loaded.predict(d)
            q = xgb_model_loaded.predict_proba(d)

            r = []
            for i in q[:, 1]:
                if i > 0.5 and i <= 0.75:
                    r.append("Need Further Evaluation")
                elif i > 0.75:
                    r.append("High Chance of Outage")
                else:
                    r.append("Non Outage")
            m = pd.DataFrame({
                "Prediction": r})
            n = pd.concat([b, m], axis=1)
            st.write(n)
            st.bar_chart(n['Prediction'].value_counts())
            col1, col2, col3 = st.columns(3)
            with col1:
                m1 = n[n['Prediction'] == "High Chance of Outage"][['Feeder',
                                                                        'Switch',
                                                                        'Section Id',
                                                                        'Pathid',
                                                                        'Lat',
                                                                        'Long',
                                                                        'Feeder_Switch_ID_V1']]
                ab_1 = m1[['Lat', 'Long']]
                ab_1.rename(columns={
                    'Lat': 'latitude',
                    'Long': 'longitude'}, inplace=True)
                st.write("High Chance of Outage")
                st.map(ab_1)
                st.write("High Chance of Outage")
                st.bar_chart(m1['Pathid'].value_counts(()))
            with col2:
                m2 = n[n['Prediction'] == "Need Further Evaluation"][['Feeder',
                                                                    'Switch',
                                                                    'Section Id',
                                                                    'Pathid',
                                                                    'Lat',
                                                                    'Long',
                                                                    'Feeder_Switch_ID_V1']]
                ab_2 = m2[['Lat', 'Long']]
                ab_2.rename(columns={
                    'Lat': 'latitude',
                    'Long': 'longitude'}, inplace=True)
                st.write("Need Further Evaluation")
                st.map(ab_2)
                st.write("Need Further Evaluation")
                st.bar_chart(m2['Pathid'].value_counts(()))

            with col3:
                m3 = n[n['Prediction'] == "Non Outage"][['Feeder',
                                                        'Switch',
                                                        'Section Id',
                                                        'Pathid',
                                                        'Lat',
                                                        'Long',
                                                        'Feeder_Switch_ID_V1']]

                ab_3 = m3[['Lat', 'Long']]
                ab_3.rename(columns={
                    'Lat': 'latitude',
                    'Long': 'longitude'}, inplace=True)
                st.write("Non Outage")
                st.map(ab_3)
                st.write("Non Outage")
                st.bar_chart(m3['Pathid'].value_counts(()))

except Exception as e:
    print(e)
