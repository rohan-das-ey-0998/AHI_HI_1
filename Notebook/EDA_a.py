import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
import pylab

st.title("EDA")
df = pd.read_csv("C:/Project/Electricity_project/GlobalUtility_EY_04052023/RawDataset/GlobalEYUtility_Final_for Demo.csv",delimiter=',')
df = df[:300]

if st.checkbox("Preview Dataset"):
    data = df.head()
    st.write(data)

if st.checkbox("Basic Observations: "):
    s = df.shape
    st.write(f"Shape of the dataset : {s}")

if st.checkbox("Basic Statistics: "):
    s = df.describe()
    st.write(s)

if st.checkbox("Percentage of Outage and Non-Outage: "):
    st.write(df.target.value_counts(normalize=True))
    st.write(df.target.value_counts(normalize=True).plot(kind='bar'))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

## Deleting some irrelevent features

try:
    df.drop(columns=['No_of_Times_Exceeded100_',
                    'No_of_Times_Exceeded90_',
                    'Peak_Slab',
                     'IR_Value_After_JointingPHASE_TO',
                     'New_Winter','_dataobs_',
                     'Feeder',
                     'Switch',
                     'Section Id',
                     'Pathid',
                    'Path Order',
                     'DATE_ONLY',
                     'Lat',
                     'Long',
                     'Customer At Risk',
                     'Revenue At Rsik',
                     'Assets At Risk',
                     'Risk Type',
                     'NoFaultsSince2016'], inplace=True)
    print("Deleted successfully")
except Exception as e:
    print(e)

cat_col = [i for i in df.columns if df[i].dtype == 'object']
num_col = [i for i in df.columns if df[i].dtype == 'float64']

if st.checkbox("Null values :"):
    n_v = st.selectbox("Select Features",("categotical","numerical"))
    if n_v == "categotical":
        st.write(df[cat_col].isna().sum())
    if n_v == "numerical":
        st.write(df[num_col].isna().sum())
    else:
        st.write("Select feature columns")


if st.checkbox("After deleting some useless features some Categorical Columns : "):
    st.write(cat_col)

if st.checkbox("After deleting some useless features some Numerical Columns : "):
    st.write(num_col)

## Select a column :
if st.checkbox("Details of some Categorical Columns : "):
    col_op = st.selectbox("Select Categorical Column",('abin',
                                                         'Feeder_Category',
                                                         'BREAKER_MAKE',
                                                         'UpcaseRoadTraffic',
                                                         'Upcase_DepthOfCable',
                                                         'Upcase_SoilCondition',
                                                         'Upcase_LeadExposed',
                                                         'Upcase_ArmourCableCondition',
                                                         'WEEKEND_Y_N_',
                                                         'New_Summer',
                                                         'New_Monsoon'))
    if col_op == 'abin':
        st.write(df.abin.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'Feeder_Category':
        st.write(df.Feeder_Category.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'BREAKER_MAKE':
        st.write(df.BREAKER_MAKE.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'UpcaseRoadTraffic':
        st.write(df.UpcaseRoadTraffic.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'Upcase_DepthOfCable':
        st.write(df.Upcase_DepthOfCable.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'Upcase_SoilCondition':
        st.write(df.Upcase_SoilCondition.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'Upcase_LeadExposed':
        st.write(df.Upcase_LeadExposed.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'Upcase_ArmourCableCondition':
        st.write(df.Upcase_ArmourCableCondition.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'WEEKEND_Y_N_':
        st.write(df.WEEKEND_Y_N_.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'New_Summer':
        st.write(df.New_Summer.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if col_op == 'New_Monsoon':
        st.write(df.New_Monsoon.value_counts().plot(kind='bar'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    else:
        st.write("Select Categorical Column")

## plotting for numerical columns
plt.style.use("ggplot")
def num_plot(df,col):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df[col].hist()
    plt.subplot(1, 2, 2)
    stat.probplot(df[col], dist='norm', plot=pylab)
    plt.title(col)
    plt.show()

## Select a column :
if st.checkbox("Details of some Numerical Columns : "):
    for i in num_col:
        num_plot(df,i)
        st.pyplot()

def biv_plot(df,col):
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.countplot(data=df,x=col, hue=df['target'])
    plt.title(col)
    plt.xticks(rotation=90)

## Select a column :
if st.checkbox("Bivariate Analysis of Categorical Columns"):
    for i in cat_col:
        biv_plot(df,i)
        st.pyplot()















