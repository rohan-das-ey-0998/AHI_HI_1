import streamlit as st
import pandas as pd

st.title("Asset Performance Management")

a = None

uploaded_files = st.file_uploader("Upload a CSV file")
if uploaded_files is not None:
    # To read file as bytes:
    bytes_data = pd.read_csv(uploaded_files)
    st.write(bytes_data)
    st.write(f"The uploaded dataset has {bytes_data.shape[0]} records and {bytes_data.shape[1]} columns.")
    a = bytes_data
    try:
        a.to_csv("raw_data\RawData_v1.csv")
        print("Dataset Saved Successfully")
    except Exception as e:
        print(e)









