import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
import pylab
import streamlit as st
import plotly as px

import plotly.graph_objects as go
import plotly.express as pl
import plotly.figure_factory as ff



st.markdown("# Feature Engineering")
st.markdown("##### Feature engineering is the process of transforming raw data into a format that is suitable for machine learning models.")

csv_files = r"C:\Project\Electricity_project\Global_Utility_05\Notebook\Original_Data\GlobalEYUtility_Final_for Demo.csv"
df = pd.read_csv(csv_files)
df_featured = df.copy()

cat_col = [i for i in df.columns if df[i].dtype == 'object']
num_col = [i for i in df.columns if df[i].dtype == 'float64']

if st.checkbox("Missing Value Imputation"):
    col1, col2 = st.columns(2)
    with col1:
        st.header("Before")
        n_v = st.selectbox("Select features before computation", ("Categotical", "Numerical"))
        if n_v == "Categotical":
            st.bar_chart(df_featured[cat_col].isna().sum())
        if n_v == "Numerical":
            st.bar_chart(df_featured[num_col].isna().sum())
    with col2:
        st.header("After")
        n_m = st.selectbox("Select features after computation",("Categotical", "Numerical"))
        if n_m == "Categotical":
            df_featured['Feeder_Category'].fillna('RES', inplace=True)
            st.bar_chart(df_featured[cat_col].isna().sum())
        if n_m == "Numerical":
            df_featured['NO__OF_JOINTS'].fillna(df_featured['NO__OF_JOINTS'].mean(), inplace=True)
            df_featured['AGE_YRS_'].fillna(df_featured['AGE_YRS_'].mean(), inplace=True)
            st.bar_chart(df_featured[num_col].isna().sum())

df['Feeder_Category'].fillna('RES', inplace=True)
df['NO__OF_JOINTS'].fillna(df['NO__OF_JOINTS'].mean(), inplace=True)
df['AGE_YRS_'].fillna(df['AGE_YRS_'].mean(), inplace=True)

if st.checkbox("Encoding categorical variables"):
    st.write("Converting categorical variables into numerical representations that machine learning algorithms can process like one-hot encoding, label encoding.")
    col3, col4 = st.columns(2)
    with col3:
        st.header("Before")
        n_3 = st.selectbox("Select Categorical Column", ('Feeder_Category',
                                                         'New_Summer',
                                                         'New_Monsoon',
                                                         'New_Winter'))

        if n_3 == 'Feeder_Category':
            st.bar_chart(df_featured['Feeder_Category'].value_counts())
        if n_3 == 'New_Summer':
            st.bar_chart(df_featured['New_Summer'].value_counts())
        if n_3 == 'New_Monsoon':
            st.bar_chart(df_featured['New_Monsoon'].value_counts())
        if n_3 == 'New_Winter':
            st.bar_chart(df_featured['New_Winter'].value_counts())

    with col4:
        st.header("After")
        n_4 = st.selectbox("Select features after computation", ('Feeder_Category',
                                                                 'New_Summer',
                                                                 'New_Monsoon',
                                                                 'New_Winter'))
        if n_4 == 'Feeder_Category':
            df_featured['Feeder_Category'] = df_featured['Feeder_Category'].map({
                'RES': 0,
                'COM': 1,
                'MIX': 2})
            st.bar_chart(df_featured['Feeder_Category'].value_counts())
        if n_4 == 'New_Summer':
            df_featured['New_Summer'] = df_featured['New_Summer'].map({
                'N': 0,
                'Y': 1})
            st.bar_chart(df_featured['New_Summer'].value_counts())
        if n_4 == 'New_Monsoon':
            df_featured['New_Monsoon'] = df_featured['New_Monsoon'].map({
                'N': 0,
                'Y': 1})
            st.bar_chart(df_featured['New_Monsoon'].value_counts())
        if n_4 == 'New_Winter':
            df_featured['New_Winter'] = df_featured['New_Winter'].map({
                'N': 0,
                'Y': 1})
            st.bar_chart(df_featured['New_Winter'].value_counts())


df['Feeder_Category'] = df['Feeder_Category'].map({
    'RES': 0,
    'COM': 1,
    'MIX': 2})
df['New_Summer'] = df['New_Summer'].map({
    'N': 0,
    'Y': 1})
df['New_Monsoon'] = df['New_Monsoon'].map({
    'N': 0,
    'Y': 1})
df['New_Winter'] = df['New_Winter'].map({
    'N': 0,
    'Y': 1})

if st.checkbox("Encoding Numerical variables"):
    st.write("Scaling numerical features to ensure that they are on a similar scale, preventing features with larger values from dominating the model's learning process i.e. standardization, end-distribution.")

    st.write("Before")
    n_3 = st.selectbox("Select Numerical Column", ('No_of_times_exceeded_60_',
                                                   'LENGTH_OF_CABLE_MTRS_',
                                                   'NO__OF_JOINTS',
                                                   'AGE_YRS_',
                                                   'Average_Current',
                                                   'Peak_Current',
                                                   'Average_voltage',
                                                   'Peak_Voltage',
                                                   'Avg_Heat_Index',
                                                   'Peak_HI',
                                                   'Avg_humidity',
                                                   'Max_Humidity',
                                                   'Avg_TEMP',
                                                   'Max_TEMP'))

    if n_3 == 'No_of_times_exceeded_60_':
        fig = go.Figure(data=[go.Histogram(x=df_featured['No_of_times_exceeded_60_'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'LENGTH_OF_CABLE_MTRS_':
        fig = go.Figure(data=[go.Histogram(x=df_featured['LENGTH_OF_CABLE_MTRS_'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'NO__OF_JOINTS':
        fig = go.Figure(data=[go.Histogram(x=df_featured['NO__OF_JOINTS'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'AGE_YRS_':
        fig = go.Figure(data=[go.Histogram(x=df_featured['AGE_YRS_'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Average_Current':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Average_Current'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Peak_Current':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Peak_Current'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Average_voltage':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Average_voltage'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Peak_Voltage':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Peak_Voltage'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Avg_Heat_Index':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Avg_Heat_Index'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Peak_HI':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Peak_HI'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Avg_humidity':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Avg_humidity'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Max_Humidity':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Max_Humidity'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Avg_TEMP':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Avg_TEMP'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_3 == 'Max_TEMP':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Max_TEMP'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)

    st.write("After")
    n_4 = st.selectbox("Select features after computation", ('No_of_times_exceeded_60_',
                                                               'LENGTH_OF_CABLE_MTRS_',
                                                               'NO__OF_JOINTS',
                                                               'AGE_YRS_',
                                                               'Average_Current',
                                                               'Peak_Current',
                                                               'Average_voltage',
                                                               'Peak_Voltage',
                                                               'Avg_Heat_Index',
                                                               'Peak_HI',
                                                               'Avg_humidity',
                                                               'Max_Humidity',
                                                               'Avg_TEMP',
                                                               'Max_TEMP'))

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
        IQR = df_featured[i].quantile(0.75) - df_featured[i].quantile(0.25)
        lower_bound = df_featured[i].quantile(0.25) - (1.5 * IQR)
        upper_bound = df_featured[i].quantile(0.75) + (1.5 * IQR)
        print(i, ":", lower_bound, ",", upper_bound)
        lower_boundries.append(lower_bound)
        upper_boundries.append(upper_bound)

    # replace the all the outliers which is greater then upper boundary by upper boundary
    j = 0
    for i in a:
        df_featured.loc[df[i] > upper_boundries[j], i] = int(upper_boundries[j])
        j = j + 1

    if n_4 == 'No_of_times_exceeded_60_':
        fig = go.Figure(data=[go.Histogram(x=df_featured['No_of_times_exceeded_60_'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'LENGTH_OF_CABLE_MTRS_':
        fig = go.Figure(data=[go.Histogram(x=df_featured['LENGTH_OF_CABLE_MTRS_'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'NO__OF_JOINTS':
        fig = go.Figure(data=[go.Histogram(x=df_featured['NO__OF_JOINTS'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'AGE_YRS_':
        fig = go.Figure(data=[go.Histogram(x=df_featured['AGE_YRS_'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Average_Current':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Average_Current'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Peak_Current':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Peak_Current'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Average_voltage':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Average_voltage'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Peak_Voltage':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Peak_Voltage'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Avg_Heat_Index':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Avg_Heat_Index'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Peak_HI':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Peak_HI'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Avg_humidity':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Avg_humidity'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Max_Humidity':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Max_Humidity'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Avg_TEMP':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Avg_TEMP'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)
    if n_4 == 'Max_TEMP':
        fig = go.Figure(data=[go.Histogram(x=df_featured['Max_TEMP'])])
        fig.update_layout(title='Histogram', xaxis=dict(title='Values'), yaxis=dict(title='Count'))
        st.plotly_chart(fig)

df['CABLE_SIZE'] = df['CABLE_SIZE'].map({
        240.00: 0,
        300.00: 1,
        225.00: 1,
        120.00: 2,
        185.00: 2,
        70.00: 2,
        0.15: 2,
        0.20: 2,
        0.30: 2})

lower_boundries = []
upper_boundries = []
a = ['No_of_times_exceeded_60_', 'LENGTH_OF_CABLE_MTRS_', 'NO__OF_JOINTS', 'AGE_YRS_',
         'Average_Current', 'Peak_Current',
         'Average_voltage', 'Peak_Voltage',
         'Avg_Heat_Index', 'Peak_HI',
         'Avg_humidity', 'Max_Humidity',
         'Avg_TEMP', 'Max_TEMP']
for i in a:
    IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
    lower_bound = df[i].quantile(0.25) - (1.5 * IQR)
    upper_bound = df[i].quantile(0.75) + (1.5 * IQR)
    print(i, ":", lower_bound, ",", upper_bound)
    lower_boundries.append(lower_bound)
    upper_boundries.append(upper_bound)

# replace the all the outliers which is greater then upper boundary by upper boundary
j = 0
for i in a:
    df.loc[df[i] > upper_boundries[j], i] = int(upper_boundries[j])
    j = j + 1

if st.checkbox("See the preprocessed data"):
    st.write(df)

try:
    df.to_csv(r"C:\Users\YR272YB\OneDrive - EY\Desktop\Projects\Utility_AHI\raw_data\Preprocessed_data_v1.csv")
    print("Preprocessed_data_v1 Dataset Saved Successfully.")
except Exception as e:
    print(e)







