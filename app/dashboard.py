import streamlit as st
import pandas as pd
import os

st.title("AI Customer Segmentation System")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(BASE_DIR, "outputs", "customer_segments.csv")

if os.path.exists(output_path):

    df = pd.read_csv(output_path)

    st.success("Segmentation Results Loaded")

    st.subheader("Customer Data")
    st.dataframe(df)

    st.subheader("Cluster Distribution")
    st.bar_chart(df["Cluster"].value_counts())

    st.subheader("Churn Risk Customers")

    if "Churn" in df.columns:
        st.dataframe(df[df["Churn"] == 1])

else:

    st.warning("No segmentation results found.")
    st.info("Run the ML pipeline to generate outputs.")