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


st.markdown("# Exploratory Data Analysis (EDA)")

csv_files = r"C:\Project\Electricity_project\Global_Utility_05\Notebook\Original_Data\GlobalEYUtility_Final_for Demo.csv"
df = pd.read_csv(csv_files)

if st.checkbox("Show the dataset"):
    st.write(df.head())

col1, col2 = st.columns(2)
with col1:
    st.header("Categorical")
    col_op = st.selectbox("Select Categorical Column",('Feeder_Category',
                                                        'New_Summer',
                                                        'New_Monsoon',
                                                        'New_Winter',
                                                        'Feeder',
                                                        'Switch',
                                                        'Section Id',
                                                        'Pathid',
                                                        'Lat',
                                                        'Long'))

    if col_op == 'Feeder_Category':
        st.bar_chart(df.Feeder_Category.value_counts())
    if col_op == 'New_Summer':
        st.bar_chart(df.New_Summer.value_counts())
    if col_op == 'New_Monsoon':
        st.bar_chart(df.New_Monsoon.value_counts())
    if col_op == 'New_Winter':
        st.bar_chart(df.New_Winter.value_counts())
    if col_op == 'Feeder':
        st.bar_chart(df.Feeder.value_counts())
    if col_op == 'Switch':
        st.bar_chart(df.Switch.value_counts())
    if col_op == 'Section Id':
        st.bar_chart(df['Section Id'].value_counts())
    if col_op == 'Pathid':
        st.bar_chart(df.Pathid.value_counts())
    if col_op == 'Lat':
        st.bar_chart(df.Lat.value_counts())
    if col_op == 'Long':
        st.bar_chart(df.Long.value_counts())

plt.style.use("ggplot")
def num_plot(df,col):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df[col].hist()
    plt.subplot(1, 2, 2)
    stat.probplot(df[col], dist='norm', plot=pylab)
    plt.title(col)
    plt.show()

with col2:
    st.header("Numerical")
    col_op = st.selectbox("Select Numerical Features", ('CABLE_SIZE',
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
    if col_op == 'CABLE_SIZE':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'CABLE_SIZE')
        st.pyplot()
    if col_op == 'LENGTH_OF_CABLE_MTRS_':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'LENGTH_OF_CABLE_MTRS_')
        st.pyplot()
    if col_op == 'NO__OF_JOINTS':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'NO__OF_JOINTS')
        st.pyplot()
    if col_op == 'AGE_YRS_':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'AGE_YRS_')
        st.pyplot()
    if col_op == 'Average_Current':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Average_Current')
        st.pyplot()
    if col_op == 'Peak_Current':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Peak_Current')
        st.pyplot()
    if col_op == 'Average_voltage':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Average_voltage')
        st.pyplot()
    if col_op == 'Peak_Voltage':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Peak_Voltage')
        st.pyplot()
    if col_op == 'Avg_Heat_Index':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Avg_Heat_Index')
        st.pyplot()
    if col_op == 'Avg_Heat_Index':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Avg_Heat_Index')
        st.pyplot()
    if col_op == 'Peak_HI':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Peak_HI')
        st.pyplot()
    if col_op == 'Avg_humidity':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Avg_humidity')
        st.pyplot()
    if col_op == 'Max_Humidity':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Max_Humidity')
        st.pyplot()
    if col_op == 'Avg_TEMP':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Avg_TEMP')
        st.pyplot()
    if col_op == 'Max_TEMP':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        num_plot(df, 'Max_TEMP')
        st.pyplot()

st.markdown("Here target is the output feature. Where 1 denotes Outage and 0 denotes Non-Outage. "
            "So, in this dataset we have around 174 Outage and 126 Non-Outage values")
if st.checkbox("Percentage of Outage and Non-Outage: "):
    st.bar_chart(df.target.value_counts(normalize=True))

cat_col = [i for i in df.columns if df[i].dtype == 'object']
num_col = [i for i in df.columns if df[i].dtype == 'float64']

if st.checkbox("Basic Statistics of the Dataset: "):
    s = df.describe()
    st.write(s)

if st.checkbox("Null values :"):
    n_v = st.selectbox("Select Features",("Categotical","Numerical"))
    if n_v == "Categotical":
        st.bar_chart(df[cat_col].isna().sum())
    if n_v == "Numerical":
        st.bar_chart(df[num_col].isna().sum())

def biv_plot(df,col):
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.countplot(data=df,x=col, hue=df['target'])
    plt.title(col)
    plt.xticks(rotation=90)

## Select a column :
if st.checkbox("Analysis of Categorical Features with Target"):
    for i in cat_col:
        biv_plot(df,i)
        st.pyplot()

if st.checkbox("Check Outliers"):
    nu_o = st.selectbox("Numerical Features", ('CABLE_SIZE',
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

    if nu_o == 'CABLE_SIZE':
        fig = pl.box(df, y="CABLE_SIZE")
        st.plotly_chart(fig,use_container_width=True)
    if nu_o == 'LENGTH_OF_CABLE_MTRS_':
        fig = pl.box(df, y="LENGTH_OF_CABLE_MTRS_")
        st.plotly_chart(fig)
    if nu_o == 'NO__OF_JOINTS':
        fig = pl.box(df, y="NO__OF_JOINTS")
        st.plotly_chart(fig)
    if nu_o == 'AGE_YRS_':
        fig = pl.box(df, y="AGE_YRS_")
        st.plotly_chart(fig)
    if nu_o == 'Average_Current':
        fig = pl.box(df, y="Average_Current")
        st.plotly_chart(fig)
    if nu_o == 'Peak_Current':
        fig = pl.box(df, y="Peak_Current")
        st.plotly_chart(fig)
    if nu_o == 'Average_voltage':
        fig = pl.box(df, y="Average_voltage")
        st.plotly_chart(fig)
    if nu_o == 'Peak_Voltage':
        fig = pl.box(df, y="Peak_Voltage")
        st.plotly_chart(fig)
    if nu_o == 'Avg_Heat_Index':
        fig = pl.box(df, y="Avg_Heat_Index")
        st.plotly_chart(fig)
    if nu_o == 'Peak_HI':
        fig = pl.box(df, y="Peak_HI")
        st.plotly_chart(fig)
    if nu_o == 'Avg_humidity':
        fig = pl.box(df, y="Avg_humidity")
        st.plotly_chart(fig)
    if nu_o == 'Max_Humidity':
        fig = pl.box(df, y="Max_Humidity")
        st.plotly_chart(fig)
    if nu_o == 'Avg_TEMP':
        fig = pl.box(df, y="Avg_TEMP")
        st.plotly_chart(fig)
    if nu_o == 'Max_TEMP':
        fig = pl.box(df, y="Max_TEMP")
        st.plotly_chart(fig)


if st.checkbox("Bivariate Analysis"):
    colm, coln = st.columns(2)
    m = None
    n = None
    with colm:
        nu_o = st.selectbox("Select Features", ('Feeder_Category',
                                                'New_Summer',
                                                'New_Monsoon',
                                                'New_Winter',
                                                'Feeder',
                                                'Switch',
                                                'Section Id',
                                                'Pathid',
                                                'Lat',
                                                'Long',
                                                'target',
                                                'CABLE_SIZE',
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
    with coln:
        nu_p = st.selectbox("Select any Features", ('Feeder_Category',
                                                    'New_Summer',
                                                    'New_Monsoon',
                                                    'New_Winter',
                                                    'Feeder',
                                                    'Switch',
                                                    'Section Id',
                                                    'Pathid',
                                                    'Lat',
                                                    'Long',
                                                    'target',
                                                    'CABLE_SIZE',
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

    fig = go.Figure(data=go.Scatter(
        x=df[nu_o],  # Replace 'x' with the column name for x-axis data
        y=df[nu_p],  # Replace 'y' with the column name for y-axis data
        mode='markers',
        marker=dict(color='blue')
    ))
    fig.update_layout(
        title='Scatter Plot',
        xaxis_title=nu_o,
        yaxis_title=nu_p,
    )
    st.plotly_chart(fig)













