import streamlit as st
import pickle
import pandas as pd

# load
file_name = "xgb_classification_v1.pkl"
xgb_model_loaded = pickle.load(open(file_name, "rb"))
print("Model Loaded Successfully")

## Prediction on
try:
    a1 = st.selectbox('Select Feeder_Category',('RES', 'COM', 'MIX'))
    st.write(a1)
    if a1 == 'RES':
        a1 = 0
    elif a1 == 'COM':
        a1 = 1
    else:
        a1 = 2
except Exception as e:
    print(e)

try:
    a2 = st.selectbox('Select CABLE_SIZE',('240.00', '300.00', '225.00','120.00','185.00',
                                       '70.00','0.15','0.20','0.30'))
    st.write(a2)
    if a2 == '240.00':
        a2 = 0
    elif a2 == '300.00' or a2 == '225.00' or a2 == '225.00':
        a2 = 1
    else:
        a2 = 2
except Exception as e:
    print(e)

try:
    a3 = st.number_input('Enter how many times current exceeds 60 : ')
    st.write(a3)
except Exception as e:
    print(e)

try:
    a4 = st.number_input('Enter Length of the cable in meters : ')
    st.write(a4)
except Exception as e:
    print(e)

try:
    a5 = st.number_input('Enter number of joints in the cable : ')
    st.write(a5)
except Exception as e:
    print(e)

try:
    a6 = st.number_input('Enter age of the cable : ')
    st.write(a6)
except Exception as e:
    print(e)

try:
    a7 = st.selectbox('If cable type is PILC ',('Yes', 'No'))
    st.write(a7)
    if a7 == 'Yes':
        a7 = 0
    else:
        a7 = 1
except Exception as e:
    print(e)

try:
    a8 = st.selectbox('If cable type is XLPE ',('Yes', 'No'))
    st.write(a8)
    if a8 == "Yes":
        a8 = 0
    else:
        a8 = 1
except Exception as e:
    print(e)

try:
    a9 = st.number_input('Enter Average Current : ')
    st.write(a9)
except Exception as e:
    print(e)

try:
    a10 = st.number_input('Enter Peak Current : ')
    st.write(a10)
except Exception as e:
    print(e)
try:
    a11 = st.number_input('Enter Average voltage : ')
    st.write(a11)
except Exception as e:
    print(e)

try:
    a12 = st.number_input('Enter Peak Voltage : ')
    st.write(a12)
except Exception as e:
    print(e)

try:
    a13 = st.number_input('Enter Avg Heat Index : ')
    st.write(a13)
except Exception as e:
    print(e)

try:
    a14 = st.number_input('Enter Peak Heat Index : ')
    st.write(a14)
except Exception as e:
    print(e)

try:
    a15 = st.number_input('Enter Average humidity : ')
    st.write(a15)
except Exception as e:
    print(e)

try:
    a16 = st.number_input('Enter Maximum humidity : ')
    st.write(a16)
except Exception as e:
    print(e)

try:
    a17 = st.number_input('Enter Average Temperature : ')
    st.write(a17)
except Exception as e:
    print(e)

try:
    a18 = st.number_input('Enter Maximum Temperature : ')
    st.write(a18)
except Exception as e:
    print(e)

try:
    a19 = st.selectbox('If it is Summer ? ',('Yes', 'No'))
    st.write(a19)
    if a19 == 'Yes':
        a19 = 0
    else:
        a19 = 1
except Exception as e:
    print(e)

try:
    a20 = st.selectbox('If it is Monsoon ? ',('Yes', 'No'))
    st.write(a20)
    if a20 == 'Yes':
        a20 = 0
    else:
        a20 = 1
except Exception as e:
    print(e)

try:
    a21 = st.selectbox('If it is Winter ? ',('Yes', 'No'))
    st.write(a21)
    if a21 == 'Yes':
        a21 = 0
    else:
        a21 = 1
except Exception as e:
    print(e)

m = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21]
st.write(m)
#a = xgb_model_loaded.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21]])
#st.write(a)

b = pd.DataFrame([m] , columns = xgb_model_loaded.feature_names_in_)
p = xgb_model_loaded.predict(b)
print(f"prediction is : {p[0]}")
if p == 0:
    st.write("Prediction is non outage")
else:
    st.write("prediction is outage")
#st.write(p[0])
#print("prediction is : {}")















