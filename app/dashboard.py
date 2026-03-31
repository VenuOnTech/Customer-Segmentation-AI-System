import streamlit as st
import pandas as pd

st.title("Customer Segmentation Dashboard")

file = st.file_uploader("Upload dataset")

if file:
    df = pd.read_csv(file)
    st.write(df.head())