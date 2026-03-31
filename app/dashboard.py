import streamlit as st
import pandas as pd
import os

st.title("Customer Segmentation Dashboard")

if os.path.exists("outputs/customer_segments.csv"):
    df = pd.read_csv("outputs/customer_segments.csv")

    st.write("Customer Segments Preview")
    st.dataframe(df.head())

    st.write("Cluster Distribution")
    st.bar_chart(df["Cluster"].value_counts())
else:
    st.warning("No output found. Run pipeline first.")