"""
Streamlit dashboard - Downloads results from GitHub Actions
"""
import streamlit as st
import pandas as pd
import os
import subprocess

st.set_page_config(
    page_title="AI Customer Segmentation System",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 AI Customer Segmentation System")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(BASE_DIR, "outputs", "customer_segments.csv")

@st.cache_resource
def download_results():
    """Download results from GitHub Release"""
    try:
        os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
        
        st.info("📥 Downloading results from GitHub...")
        
        # Download from GitHub Release (update with your latest release URL)
        subprocess.run([
            "curl", "-L", "-o", output_path,
            "https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases/download/v1.0.0-outputs/customer_segments.csv"
        ], check=True)
        
        st.success("✅ Results downloaded!")
        return pd.read_csv(output_path)
        
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        return None

# Try to load results
if os.path.exists(output_path):
    df = pd.read_csv(output_path)
else:
    df = download_results()

if df is not None and len(df) > 0:
    
    st.success("✅ Segmentation Results Loaded")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Number of Segments", df["Cluster"].nunique())
    with col3:
        if "Churn" in df.columns:
            st.metric("At-Risk Customers", (df["Churn"] == 1).sum())
    
    st.divider()
    
    # Display customer data
    st.subheader("📋 Customer Data")
    st.dataframe(df, use_container_width=True)
    
    # Cluster distribution
    if "Cluster" in df.columns:
        st.subheader("📈 Cluster Distribution")
        cluster_counts = df["Cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)
    
    # Churn analysis
    if "Churn" in df.columns:
        st.subheader("⚠️ Churn Risk Customers")
        churn_df = df[df["Churn"] == 1]
        if len(churn_df) > 0:
            st.dataframe(churn_df, use_container_width=True)
            st.info(f"Total at-risk customers: {len(churn_df)}")
        else:
            st.info("No churn risk customers detected")

else:
    st.error("❌ Failed to load segmentation data")
    st.info("Run the ML pipeline to generate results: `python main.py`")