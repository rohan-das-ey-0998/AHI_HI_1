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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropDuplicateFeatures
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, recall_score, precision_score, f1_score

from keras.models import Sequential
from keras.layers import Dense,Dropout


st.markdown("# ML Methods")
st.markdown("##### This section talks about different Machine Learning Algorithms and Neural Network techniques.")
csv_files = r"C:\Users\YR272YB\OneDrive - EY\Desktop\Projects\Utility_AHI\raw_data\Preprocessed_data_v1.csv"
df = pd.read_csv(csv_files)
df = df[['Feeder_Category',
         'CABLE_SIZE',
         'No_of_times_exceeded_60_',
         'LENGTH_OF_CABLE_MTRS_',
         'NO__OF_JOINTS',
         'AGE_YRS_',
         'PILC',
         'XLPE',
         'DATE_ONLY',
         'Average_Current',
         'Peak_Current',
         'Average_voltage',
         'Peak_Voltage',
         'target',
         'Avg_Heat_Index',
         'Peak_HI',
         'Avg_humidity',
         'Max_Humidity',
         'Avg_TEMP',
         'Max_TEMP',
         'New_Summer',
         'New_Monsoon',
         'New_Winter',
         'Feeder',
         'Switch',
         'Section Id',
         'Pathid',
         'Lat',
         'Long',
         'Feeder_Switch_ID_V1']]

df_featured = df.copy()
print(df.columns)

df_new = df.drop(columns=['DATE_ONLY','Section Id','Pathid','Feeder_Switch_ID_V1','Lat','Long'])
df_featured_new = df_featured.drop(columns=['DATE_ONLY','Section Id','Pathid','Feeder_Switch_ID_V1','Lat','Long'])
st.write(df_new)

X_1 = df_featured_new.drop(columns=['target'])
y_1 = df_featured_new['target']

X_train, X_test, y_train, y_test = train_test_split(X_1,y_1,test_size=0.2,random_state=100,stratify=y_1.values)
if st.checkbox("Proportion of Outage and Non-Outage in train and test data"):
    col1,col2 = st.columns(2)
    with col1:
        st.write("Training Set")
        st.bar_chart(y_train.value_counts(normalize=True))
    with col2:
        st.write("Testing Set")
        st.bar_chart(y_test.value_counts(normalize=True))

if st.checkbox("Logistic Regression"):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)
    col3, col4 = st.columns(2)
    with col3:
        st.header('Training:')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_train, y_train_hat),
            'Precision': precision_score(y_train, y_train_hat),
            'Recall': recall_score(y_train, y_train_hat),
            'F1' : f1_score(y_train, y_train_hat)
        },index=['Score']))
    with col4:
        st.header('Testing:')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_test, y_test_hat),
            'Precision': precision_score(y_test, y_test_hat),
            'Recall': recall_score(y_test, y_test_hat),
            'F1': f1_score(y_test, y_test_hat)
        }, index=['Score']))

if st.checkbox("Decision Tree Classifier"):
    model_2 = DecisionTreeClassifier()
    model_2.fit(X_train, y_train)
    y_train_hat = model_2.predict(X_train)
    y_test_hat = model_2.predict(X_test)
    col5, col6 = st.columns(2)
    with col5:
        st.header('Training:')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_train, y_train_hat),
            'Precision': precision_score(y_train, y_train_hat),
            'Recall': recall_score(y_train, y_train_hat),
            'F1' : f1_score(y_train, y_train_hat)
        },index=['Score']))
    with col6:
        st.header('Testing:')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_test, y_test_hat),
            'Precision': precision_score(y_test, y_test_hat),
            'Recall': recall_score(y_test, y_test_hat),
            'F1': f1_score(y_test, y_test_hat)
        }, index=['Score']))

if st.checkbox("Random Forest Classifier"):
    model_3 = RandomForestClassifier()
    model_3.fit(X_train, y_train)
    y_train_hat = model_3.predict(X_train)
    y_test_hat = model_3.predict(X_test)
    col7, col8 = st.columns(2)
    with col7:
        st.header('Training:')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_train, y_train_hat),
            'Precision': precision_score(y_train, y_train_hat),
            'Recall': recall_score(y_train, y_train_hat),
            'F1' : f1_score(y_train, y_train_hat)
        },index=['Score']))
    with col8:
        st.header('Testing:')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_test, y_test_hat),
            'Precision': precision_score(y_test, y_test_hat),
            'Recall': recall_score(y_test, y_test_hat),
            'F1': f1_score(y_test, y_test_hat)
        }, index=['Score']))

if st.checkbox("XGBoost Classifier"):
    model_4 = XGBClassifier()
    model_4.fit(X_train, y_train)
    y_train_hat = model_4.predict(X_train)
    y_test_hat = model_4.predict(X_test)
    col9, col10 = st.columns(2)
    with col9:
        st.header('Training:')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_train, y_train_hat),
            'Precision': precision_score(y_train, y_train_hat),
            'Recall': recall_score(y_train, y_train_hat),
            'F1' : f1_score(y_train, y_train_hat)
        },index=['Score']))
    with col10:
        st.header('Testing:')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_test, y_test_hat),
            'Precision': precision_score(y_test, y_test_hat),
            'Recall': recall_score(y_test, y_test_hat),
            'F1': f1_score(y_test, y_test_hat)
        }, index=['Score']))

if st.checkbox("Neural Network"):
    from keras import backend as K
    model_5 = Sequential()
    model_5.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model_5.add(Dense(64, activation='relu'))
    model_5.add(Dense(128, activation='relu'))
    model_5.add(Dense(256, activation='relu'))
    model_5.add(Dropout(0.4))
    model_5.add(Dense(128, activation='relu'))
    model_5.add(Dense(1, activation='sigmoid'))

    # Functions to calculate precision recall
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    # Compile the model
    model_5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', precision_m, recall_m, f1_m])
    model_5.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    col11,col12 = st.columns(2)
    with col11:
        st.header("Training")
        # loss, accuracy = model_5.evaluate(X_train, y_train)
        loss, accuracy, precision, recall, f1_score = model_5.evaluate(X_train, y_train, verbose=0)

        st.write(pd.DataFrame({
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1_score
        },index=["Score"]))

    with col12:
        st.header("Testing")
        # loss, accuracy = model_5.evaluate(X_train, y_train)
        loss_1, accuracy_1, precision_1, recall_1, f1_score_1 = model_5.evaluate(X_test, y_test, verbose=0)

        st.write(pd.DataFrame({
            "Accuracy": accuracy_1,
            "Precision": precision_1,
            "Recall": recall_1,
            "F1": f1_score_1
        },index=["Score"]))












