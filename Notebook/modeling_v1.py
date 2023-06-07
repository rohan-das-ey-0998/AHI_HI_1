import streamlit as st
import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stat
import pylab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

st.title("Modeling")
df = pd.read_csv("C:/Project/Electricity_project/GlobalUtility_EY_04052023/RawDataset/GlobalEYUtility_Final_for Demo.csv",delimiter=',')
df = df[:300]

if st.checkbox("Preview Dataset"):
    data = df.head()
    st.write(data)

if st.checkbox("Percentage of Outage and Non-Outage: "):
    st.write(df.target.value_counts(normalize=True))
    st.write(df.target.value_counts(normalize=True).plot(kind='bar'))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

try:
    df.drop(columns=['No_of_Times_Exceeded100_',
                     'MORNING_AVG_CURRENT',
                     'AFTERNOON_AVG_CURRENT',
                     'EVENING_AVG_CURRENT',
                     'MORNING_AVG_VOLTAGE',
                     'AFTERNOON_AVG_VOLTAGE',
                     'EVENING_AVG_VOLTAGE',
                     'Morning_AVG_Heat_Index',
                     'Afternoon_AVG_Heat_Index',
                     'Evening_Avg_Heat_Index',

                     'Morning_AVG_humidity',
                     'Afternoon_AVG_HUMIDITY',
                     'Evening_Avg_HUMIDITY',

                     'Morning_AVG_TEMP',
                     'Afternoon_AVG_TEMP',
                     'Evening_Avg_TEMP',

                     '_5Yr_Faults_Count',
                     'FaultCurrent_Count',
                     'No_of_Times_Exceeded90_',
                     'No_of_time_below_60_',
                     'Peak_Slab',
                     'abin',
                     'BREAKER_MAKE',
                     'DATE_ONLY',
                     'DAY_OF_WEEK',
                     'WEEKEND_Y_N_',
                     'Upcase_DepthOfCable',
                     'NO__OF_PARALLEL_RUNNING_CABLES',
                     'IR_Value_After_JointingPHASE_TO',
                     'MORNING_AVG_CURRENT_N_1',
                     'MORNING_AVG_CURRENT_N_2',
                     'AFTERNOON_AVG_CURRENT_N_1',
                     'AFTERNOON_AVG_CURRENT_N_2',
                     'EVENING_AVG_CURRENT_N_1',
                     'EVENING_AVG_CURRENT_N_2',
                     'MORNING_AVG_VOLTAGE_N_1',
                     'MORNING_AVG_VOLTAGE_N_2',
                     'AFTERNOON_AVG_VOLTAGE_N_1',
                     'AFTERNOON_AVG_VOLTAGE_N_2',
                     'EVENING_AVG_VOLTAGE_N_1',
                     'EVENING_AVG_VOLTAGE_N_2',
                     'TARGET_N_1',
                     'TARGET_N_2',
                     'AVERAGE_CURRENT_N_1',
                     'AVERAGE_CURRENT_N_2',
                     'AVERAGE_VOLTAGE_N_1',
                     'AVERAGE_VOLTAGE_N_2',
                     'AVERAGE_HEAT_INDEX_N_1',
                     'AVERAGE_HEAT_INDEX_N_2',
                     'AVERAGE_HUMUDITY_N_1',
                     'AVERAGE_HUMUDITY_N_2',
                     'AVERAGE_TEMP_N_1',
                     'AVERAGE_TEMP_N_2',
                     'MORNING_AVG_HEAT_INDEX_N_1',
                     'MORNING_AVG_HEAT_INDEX_N_2',
                     'AFTERNOON_AVG_HEAT_INDEX_N_1',
                     'AFTERNOON_AVG_HEAT_INDEX_N_2',
                     'EVENING_AVG_HEAT_INDEX_N_1',
                     'EVENING_AVG_HEAT_INDEX_N_2',
                     'MORNING_AVG_HUMUDITY_N_1',
                     'MORNING_AVG_HUMUDITY_N_2',
                     'AFTERNOON_AVG_HUMUDITY_N_1',
                     'AFTERNOON_AVG_HUMUDITY_N_2',
                     'EVENING_AVG_HUMUDITY_N_1',
                     'EVENING_AVG_HUMUDITY_N_2',
                     'MORNING_AVG_TEMP_N_1',
                     'MORNING_AVG_TEMP_N_2',
                     'AFTERNOON_AVG_TEMP_N_1',
                     'AFTERNOON_AVG_TEMP_N_2',
                     'EVENING_AVG_TEMP_N_1',
                     'EVENING_AVG_TEMP_N_2',
                     'Peak_Current_N_1',
                     'Peak_Current_N_2',
                     'Peak_Current_Time_N_1',
                     'Peak_Current_Time_N_2',
                     'Peak_Current_Slab_N_1',
                     'Peak_Current_Slab_N_2',
                     'New_PublicHoliday',
                     '_dataobs_',
                     'Feeder',
                     'Switch',
                     'Upcase_LeadExposed',
                     'Section Id',
                     'Pathid',
                     'Path Order',
                     'Lat',
                     'Long',
                     'No_of_Load_transfer_Operations',
                     'Number_OF_Cables',
                     'Customer At Risk',
                     'Revenue At Rsik',
                     'Assets At Risk',
                     'Risk Type',
                     'NoFaultsSince2016',
                     'UpcaseRoadTraffic', 'Upcase_SoilCondition', 'Upcase_ArmourCableCondition'], inplace=True)
    print("Deleted successfully")
except Exception as e:
    print(e)

if st.checkbox("After deleting some features-> final dataset : "):
    st.write(df.head())

cat_col = [i for i in df.columns if df[i].dtype == 'object']
num_col = [i for i in df.columns if df[i].dtype == 'float64']

df['Feeder_Category'].fillna('RES',inplace=True)
df['NO__OF_JOINTS'].fillna(df['NO__OF_JOINTS'].mean(),inplace=True)
df['AGE_YRS_'].fillna(df['AGE_YRS_'].mean(),inplace=True)

## Feeder_Category
df['Feeder_Category'] = df['Feeder_Category'].map({
                                                    'RES':0,
                                                    'COM':1,
                                                    'MIX':2})

df['New_Summer'] = df['New_Summer'].map({
                                        'N':0,
                                        'Y':1})
df['New_Monsoon'] = df['New_Monsoon'].map({
                                        'N':0,
                                        'Y':1})
df['New_Winter'] = df['New_Winter'].map({
                                        'N':0,
                                        'Y':1})

## CABLE_SIZE
df['CABLE_SIZE'] = df['CABLE_SIZE'].map({
    240.00 : 0,
    300.00 : 1,
    225.00 : 1,
    120.00: 2,
    185.00 : 2,
    70.00 : 2,
    0.15 : 2,
    0.20: 2,
    0.30 : 2})

if st.checkbox("After Preprocessing and all : "):
    st.write(df.head())
    st.write(df.columns)

a = ['No_of_times_exceeded_60_','LENGTH_OF_CABLE_MTRS_','NO__OF_JOINTS','AGE_YRS_',
     'Average_Current','Peak_Current',
     'Average_voltage', 'Peak_Voltage',
     'Avg_Heat_Index', 'Peak_HI',
     'Avg_humidity', 'Max_Humidity',
     'Avg_TEMP','Max_TEMP']

# compute interquantile range to calculate the boundaries
lower_boundries = []
upper_boundries = []
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

if st.checkbox("Correlation Matrix :  "):
    #plt.figure(figsize=(20, 16))
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True,ax=ax)
    st.write(ax)
    st.pyplot()

X = df.drop(columns='target')
X_new = X.copy()
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
if st.checkbox("Proportion of Outage and Non-Outage in train and test data"):
    st.write("Training : ",y_train.value_counts(normalize=True))
    st.write("Test : ", y_test.value_counts(normalize=True))

## creating a pipeline
algos = [LogisticRegression(),
         DecisionTreeClassifier(),
         RandomForestClassifier(),
         XGBClassifier()]
names = ["Logistic Regression",
         "DecisionTree Classifier",
         "RandomForest Classifier",
         "XGBClassifier"]
acc = []
pre = []
rec = []
f1 = []
r_a = []

for name in algos:
    model = name
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    reca = recall_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    acc.append(accu)
    pre.append(prec)
    rec.append(rec)
    f1.append(f1_score)
    r_a.append(roc_auc)

if st.checkbox("Different Model Scores : "):
    st.write(pd.DataFrame({'Name':names,
                           'Accuracy_Score':acc,
                          'Precision' : pre,
                          'Recall' : reca,
                          'F_1 Score ': f1_sc,
                          'ROC_AUC' : r_a}))

## odds ratio for Logistic regression
lr = LogisticRegression(penalty='none',C=1.0)
lr.fit(X_train,y_train)
odds_ratio=np.exp(lr.coef_)
a = [i for i in odds_ratio[0]]
b = lr.feature_names_in_.tolist()
m = pd.DataFrame({
    'Features':b,
    'Odds Ratio':a})
m['Probabilty_Faulty'] = m['Odds Ratio']/(1+m['Odds Ratio'])
m['Probabilty_Non_Faulty'] = 1 - m['Probabilty_Faulty']
if st.checkbox("Odds Ratio Based on Logistic Regression : "):
    st.write(m)

# Decision Tree Plot
dt = DecisionTreeClassifier(criterion='gini',
                           max_depth=10,
                           max_features='sqrt',
                           min_samples_leaf=1,
                           min_samples_split=12)
model = dt.fit(X_train,y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

if st.checkbox("Decision Tree Plot : "):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.figure(figsize=(50, 40))
    _ = tree.plot_tree(model,
                       feature_names=X.columns,
                       class_names=y_train.values.astype(str),
                       filled=True)
    st.pyplot()

## prediction part
## As XGboost gives best result

xgb_model = XGBClassifier(random_state=123)
xgb_model.fit(X_train, y_train)
y_train_hat = xgb_model.predict(X_train)
y_test_hat = xgb_model.predict(X_test)
print(xgb_model)
print('Train performance')
print('-------------------------------------------------------')
print(classification_report(y_train, y_train_hat))
print('Test performance')
print('-------------------------------------------------------')
print(classification_report(y_test, y_test_hat))
print('Roc_auc score')
print('-------------------------------------------------------')
print(roc_auc_score(y_test, y_test_hat))
print('')
print('Confusion matrix')
print('-------------------------------------------------------')
print(confusion_matrix(y_test, y_test_hat))

import pickle
file_name = "xgb_classification_v1.pkl"
# save
try:
    pickle.dump(xgb_model, open(file_name, "wb"))
    print("Model File saved Successfully")
except Exception as e:
    print(e)

## prediction part based on model saved
# load
file_name = "xgb_classification_v1.pkl"
xgb_model_loaded = pickle.load(open(file_name, "rb"))
## Prediction on Test set

a = xgb_model_loaded.predict(X_train)
print(a)
print(pd.DataFrame(list(zip(y_test,a))))

