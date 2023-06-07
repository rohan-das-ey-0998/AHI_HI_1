import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
import pylab
import streamlit as st
import plotly as px
import klib
import plotly.graph_objects as go
import plotly.express as pl
import plotly.figure_factory as ff


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropDuplicateFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SequentialFeatureSelector

st.markdown("# Feature Selection")
st.markdown("##### Feature selection is an important step in machine learning to identify the most relevant and informative features for training models.")
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

df_new = df.drop(columns=['DATE_ONLY','Section Id','Pathid','Feeder_Switch_ID_V1'])
df_featured_new = df_featured.drop(columns=['DATE_ONLY','Section Id','Pathid','Feeder_Switch_ID_V1'])

X_1 = df_featured_new.drop(columns=['target'])
y_1 = df_featured_new['target']
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1,y_1,test_size=0.2,random_state=0)

if st.checkbox("Constant Features"):
    vt = VarianceThreshold(threshold=0)  ## i.e. variance=0
    vt.fit(X_train_1)
    st.write(X_train_1.columns[~vt.get_support()])

if st.checkbox("Quasi Constant i.e. same values in all observations"):
    vt_1 = VarianceThreshold(threshold=0.05)  ## i.e. variance=0
    vt_1.fit(X_train_1)
    st.write(X_train_1.columns[~vt_1.get_support()])

if st.checkbox("Duplicated Features"):
    du_1 = DropDuplicateFeatures(variables=None, missing_values='raise')
    du_1.fit(X_train_1)
    st.write(du_1.features_to_drop_)

def correlation(dataset, threshold):
    col_corr = set()
    corr_mat = dataset.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i,j]) > threshold:
                colname = corr_mat.columns[i]
                col_corr.add(colname)
    return col_corr

if st.checkbox("Correlated Features"):
    st.write("Positive correlation")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    klib.corr_plot(df_featured_new,split="pos")
    # Render the plot in Streamlit
    st.pyplot()

    st.write("Negative correlation")
    klib.corr_plot(df_featured_new, split="neg")
    # Render the plot in Streamlit
    st.pyplot()

if st.checkbox("Embedded Methods"):
    if st.checkbox("L1 regularization or Lasso"):
        lasso = Lasso(alpha=0.1)  # Alpha controls the regularization strength
        lasso.fit(X_train_1, y_train_1)

        importance_scores = abs(lasso.coef_)
        feature_names = X_train_1.columns
        feature_importance = dict(zip(feature_names, importance_scores))

        sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        st.write(pd.DataFrame(sorted_feature_importance))
    if st.checkbox("Random Forest - Tree-based Methods"):
        rf = RandomForestClassifier(n_estimators=150, random_state=40, max_depth=6)
        rf.fit(X_train_1, y_train_1)
        rf_imp = pd.concat([pd.Series(X_train_1.columns), pd.Series(rf.feature_importances_).abs()], axis=1)
        rf_imp.columns = ['Features', 'Importance']
        st.write(rf_imp.sort_values(by='Importance', ascending=False))

if st.checkbox("Wrapper Methods"):
    if st.checkbox("Sequestial Forward Feature selection"):
        sfs = SequentialFeatureSelector(estimator=RandomForestClassifier(),
                                        n_features_to_select=20,
                                        direction='forward',
                                        scoring='roc_auc',
                                        cv=2,
                                        n_jobs=4)

        sfs.fit(X_train_1, y_train_1)
        st.write(sfs.get_feature_names_out())
        print(sfs.get_feature_names_out())
    if st.checkbox("Sequential Backward Feature selection"):
        sfs_b = SequentialFeatureSelector(estimator=RandomForestClassifier(),
                                          n_features_to_select=10,
                                          direction='backward',
                                          scoring='roc_auc',
                                          cv=2,
                                          n_jobs=4)
        sfs_b.fit(X_train_1, y_train_1)
        st.write(sfs_b.get_feature_names_out())
        print(sfs_b.get_feature_names_out())


































