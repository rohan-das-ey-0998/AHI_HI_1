import streamlit as st
import pandas as pd
pd.set_option('display.max_columns', None)
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy import stats
import pylab
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,auc,roc_curve
import plotly as px

import plotly.graph_objects as go
import plotly.express as pl
import plotly.figure_factory as ff
from streamlit_shap import st_shap
import shap
from sklearn.ensemble import RandomForestClassifier
from interpret.glassbox import LogisticRegression,ClassificationTree, ExplainableBoostingClassifier
from interpret import show
from shapash import SmartExplainer


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
         'New_Winter',
         'target']]
X_1 = df.drop(columns=['target'])
y_1 = df['target']
X_train, X_test, y_train, y_test = train_test_split(X_1,y_1,test_size=0.2,random_state=100,stratify=y_1.values)

st.markdown("# Observations")
if st.checkbox("Preview Dataset"):
    data = df.head()
    st.write(data)

if st.checkbox("Results Summary"):
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
        rec.append(reca)
        f1.append(f1_sc)
        r_a.append(roc_auc)
    st.write(pd.DataFrame({'Name': names,
                           'Accuracy_Score': acc,
                           'Precision': pre,
                           'Recall': rec,
                           'F_1 Score ': f1,
                           'ROC_AUC': r_a}))

xgb_model = XGBClassifier(n_estimators=100,
                              max_depth=4,
                              eta=0.1)
xgb_model.fit(X_train, y_train)

if st.checkbox("Best Results"):
    y_train_hat = xgb_model.predict(X_train)
    y_test_hat = xgb_model.predict(X_test)
    print(xgb_model)
    col1,col2 = st.columns(2)
    with col1:
        st.write('Training')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_train, y_train_hat),
            'Precision': precision_score(y_train, y_train_hat),
            'Recall': recall_score(y_train, y_train_hat),
            'F1': f1_score(y_train, y_train_hat),
            'ROC_AUC': roc_auc_score(y_train, y_train_hat)}, index=['Score']))
        cf_matrix = confusion_matrix(y_train, y_train_hat)
        TN = cf_matrix[0][0]
        FN = cf_matrix[1][0]
        TP = cf_matrix[1][1]
        FP = cf_matrix[0][1]
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ax.xaxis.set_ticklabels(['False', 'True'])
        ax.yaxis.set_ticklabels(['False', 'True'])
        plt.show()
        st.pyplot()
        fpr, tpr, thresholds = roc_curve(y_train, y_train_hat)
        auc_score = roc_auc_score(y_train, y_train_hat)
        fig = go.Figure(data=go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name='ROC curve (AUC = {:.2f})'.format(auc_score)))
        fig.add_shape(
            type='line',
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color='black', dash='dash'))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',)
        st.plotly_chart(fig)
    with col2:
        st.write('Testing')
        st.write(pd.DataFrame({
            'Accuracy': accuracy_score(y_test, y_test_hat),
            'Precision': precision_score(y_test, y_test_hat),
            'Recall': recall_score(y_test, y_test_hat),
            'F1': f1_score(y_test, y_test_hat),
            'ROC_AUC': roc_auc_score(y_test, y_test_hat)}, index=['Score']))
        cf_matrix = confusion_matrix(y_test, y_test_hat)
        TN = cf_matrix[0][0]
        FN = cf_matrix[1][0]
        TP = cf_matrix[1][1]
        FP = cf_matrix[0][1]
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ax.xaxis.set_ticklabels(['False', 'True'])
        ax.yaxis.set_ticklabels(['False', 'True'])
        plt.show()
        st.pyplot()
        fpr, tpr, thresholds = roc_curve(y_test, y_test_hat)
        auc_score = roc_auc_score(y_test, y_test_hat)
        fig = go.Figure(data=go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name='ROC curve (AUC = {:.2f})'.format(auc_score)))
        fig.add_shape(
            type='line',
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color='black', dash='dash'))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate', )
        st.plotly_chart(fig)

file_name = r"C:\Users\YR272YB\OneDrive - EY\Desktop\Projects\Utility_AHI\model_store\xgb_classification_v2.pkl"
## save
try:
    pickle.dump(xgb_model, open(file_name, "wb"))
    print("Model File saved Successfully")
except Exception as e:
    print(e)

if st.checkbox("Feature Contributions"):
    rf_1 = RandomForestClassifier()
    rf_1.fit(X_train, y_train)
    y_pred = rf_1.predict(X_test)
    explainer = shap.TreeExplainer(rf_1)
    start_index = 1
    end_index = 2

    shap_values = explainer.shap_values(X_test[start_index:end_index])
    shap.initjs()
    st_shap(shap.force_plot(explainer.expected_value[1],
                    shap_values[1],
                    X_test[start_index:end_index]))

    ## Shapash library
    y_pred_shapash_1 = pd.DataFrame(rf_1.predict(X_test), columns=['pred'], index=X_test.index)
    xpl = SmartExplainer(model=rf_1)
    xpl.compile(x=X_test,
                y_pred=y_pred_shapash_1,
                y_target=y_test  # Optional: allows to display True Values vs Predicted Values
    )

    app_shapash = xpl.run_app(title_story='Feature Contributions')
    st.write("Detailed View : ' http://10.216.17.57:8050")

if st.checkbox("Prediction on test set"):
    st.write("Test Set")
    st.write(X_test)
    y_pred_1=xgb_model.predict(X_test)
    y_pred_prob_1 = xgb_model.predict_proba(X_test)
    st.write("Prediction")
    st.write(pd.DataFrame(list(zip(y_test,y_pred_prob_1[:,0],y_pred_prob_1[:,1],y_pred_1)),columns=['Actual_Outcome','Probabilaty_0','Probabilaty_1','Prediction_Outcome']))