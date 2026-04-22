import streamlit as st
import pandas as pd
import os
import requests
from datetime import datetime
import plotly.express as px

st.set_page_config(
    page_title="AI Customer Segmentation System",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 AI Customer Segmentation System")
st.markdown("### 🚀 Autonomous ML Pipeline with Explainable AI + Drift Monitoring")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(BASE_DIR, "outputs", "customer_segments.csv")


# ==============================
# LOAD DATA (SMART LOADER 🔥)
# ==============================

def load_data():

    # 1️⃣ Try local file
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            st.success("✅ Loaded local pipeline output")
            return df
        except:
            pass

    # 2️⃣ Try GitHub release
    try:
        st.info("📥 Fetching latest data from GitHub...")

        url = "https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases/download/latest/customer_segments.csv"

        response = requests.get(url)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)

            df = pd.read_csv(output_path)
            st.success("✅ Loaded GitHub release data")
            return df

    except Exception as e:
        st.warning(f"⚠️ GitHub fetch failed: {str(e)}")

    return None


df = load_data()


# ==============================
# DASHBOARD
# ==============================

if df is not None and len(df) > 0:

    st.success("✅ Segmentation Results Loaded")

    cluster_col = "Final_Cluster" if "Final_Cluster" in df.columns else None

    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("📊 Customers", f"{len(df):,}")
    col2.metric("🎯 Segments", df[cluster_col].nunique() if cluster_col else "N/A")
    col3.metric("⚠️ At Risk", int((df["Churn"] == 1).sum()) if "Churn" in df else "N/A")
    col4.metric("📈 Avg Purchase", f"{df['Purchase_Probability'].mean():.2%}" if "Purchase_Probability" in df else "N/A")
    col5.metric("⏱️ Avg Recency", f"{df['Recency'].mean():.1f}" if "Recency" in df else "N/A")

    st.divider()

    if cluster_col and "Frequency" in df.columns and "Monetary" in df.columns:

        fig = px.scatter(
            df.sample(n=min(5000, len(df))),
            x="Frequency",
            y="Monetary",
            color=cluster_col,
            title="Customer Segments"
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("🧠 System Status")

    colA, colB, colC = st.columns(3)

    colA.success("Clustering Active")
    colB.success("Explainability Active" if "Explanation" in df else "Explainability Missing")
    colC.success("Drift Monitoring Active")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data",
        "📈 Clusters",
        "⚠️ Churn",
        "🔍 Insights"
    ])

    with tab1:
        st.dataframe(df, width="stretch")

    with tab2:
        if cluster_col:
            st.bar_chart(df[cluster_col].value_counts())

    with tab3:
        if "Churn" in df:
            st.dataframe(df[df["Churn"] == 1])

    with tab4:
        if "Explanation" in df and df["Explanation"].notnull().any():
            sample = df.sample(n=min(10, len(df)))

            for _, row in sample.iterrows():
                if row["Explanation"] not in ["", "Not computed"]:
                    st.write(f"Customer {int(row.get('CustomerID', 0))} → {row['Explanation']}")

else:

    st.error("❌ No data available")

    st.info("""
    Fix:
    1. Run pipeline → python main.py  
    2. Upload outputs/customer_segments.csv to GitHub Release  
    3. Refresh dashboard  
    """)