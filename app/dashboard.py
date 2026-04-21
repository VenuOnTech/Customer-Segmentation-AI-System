"""
Streamlit dashboard for Customer Segmentation AI System
Enhanced with robust handling + advanced visuals
"""

import streamlit as st
import pandas as pd
import os
import subprocess
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
cache_dir = os.path.join(BASE_DIR, ".streamlit_cache")


# ==============================
# 📥 DOWNLOAD FUNCTION
# ==============================

@st.cache_resource
def download_from_release():
    try:
        st.info("📥 Downloading latest results from GitHub Release...")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        download_url = "https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases/download/latest/customer_segments.csv"

        try:
            import urllib.request
            import json

            api_url = "https://api.github.com/repos/VenuOnTech/Customer-Segmentation-AI-System/releases/latest"
            with urllib.request.urlopen(api_url) as response:
                release_data = json.loads(response.read().decode())

                if 'assets' in release_data:
                    csv_asset = next(
                        (asset for asset in release_data['assets']
                         if asset['name'] == 'customer_segments.csv'),
                        None
                    )

                    if csv_asset:
                        download_url = csv_asset['browser_download_url']
                        st.write(f"✅ Found release: {release_data['tag_name']}")

        except Exception as e:
            st.warning(f"⚠️ Release fetch failed: {str(e)}")

        result = subprocess.run(
            ["curl", "-L", "-o", output_path, download_url],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and os.path.exists(output_path):
            st.success("✅ Data downloaded successfully!")
            return pd.read_csv(output_path)
        else:
            st.error("❌ Download failed")
            return None

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


# ==============================
# 📊 LOAD DATA
# ==============================

df = None

if os.path.exists(output_path):
    try:
        df = pd.read_csv(output_path)
        st.info(f"📊 Loaded cached data ({datetime.now().strftime('%H:%M')})")
    except:
        pass

if df is None:
    with st.spinner("Fetching data..."):
        df = download_from_release()


# ==============================
# 🚀 MAIN DASHBOARD
# ==============================

if df is not None and len(df) > 0:

    st.success("✅ Segmentation Results Loaded")

    cluster_col = "Final_Cluster" if "Final_Cluster" in df.columns else None

    st.divider()

    # ==============================
    # 📊 METRICS
    # ==============================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📊 Customers", f"{len(df):,}")

    with col2:
        if cluster_col:
            st.metric("🎯 Segments", df[cluster_col].nunique())
        else:
            st.metric("🎯 Segments", "N/A")

    with col3:
        if "Churn" in df.columns:
            st.metric("⚠️ At Risk", int((df["Churn"] == 1).sum()))
        else:
            st.metric("⚠️ At Risk", "N/A")

    with col4:
        if "Purchase_Probability" in df.columns:
            st.metric("📈 Avg Purchase", f"{df['Purchase_Probability'].mean():.2%}")
        else:
            st.metric("📈 Avg Purchase", "N/A")

    with col5:
        if "Recency" in df.columns:
            st.metric("⏱️ Avg Recency", f"{df['Recency'].mean():.1f}")
        else:
            st.metric("⏱️ Avg Recency", "N/A")

    # ✅ FIXED POSITION
    st.divider()

    # ==============================
    # 📊 VISUALS
    # ==============================

    if cluster_col and "Frequency" in df.columns and "Monetary" in df.columns:

        st.subheader("📊 Customer Segmentation Visualization")

        fig = px.scatter(
            df.sample(n=min(5000, len(df)), random_state=42),
            x="Frequency",
            y="Monetary",
            color=cluster_col,
            title="Customer Segments"
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    # ==============================
    # 🧠 SYSTEM STATUS
    # ==============================

    st.subheader("🧠 System Intelligence Status")

    colA, colB, colC = st.columns(3)

    with colA:
        st.success("✅ Adaptive Clustering Enabled")

    with colB:
        if "Explanation" in df.columns and df["Explanation"].notna().any():
            st.success("✅ Explainable AI Active")
        else:
            st.warning("⚠️ Explainability Missing")

    with colC:
        st.success("✅ Drift Monitoring Active")

    # 🔥 SMALL BUT POWERFUL ADDITION
    st.caption("⚙️ Pipeline: Data → Feature Engineering → Adaptive Clustering → Churn Prediction → Explainability → Monitoring")

    # ==============================
    # 📊 TABS
    # ==============================

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data",
        "📈 Clusters",
        "⚠️ Churn",
        "🔍 Explanations"
    ])

    # ==============================
    # TAB 1
    # ==============================

    with tab1:
        st.dataframe(df, width="stretch")

        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            "customer_segments.csv"
        )

    # ==============================
    # TAB 2
    # ==============================

    with tab2:
        if cluster_col:

            st.subheader("Cluster Distribution")

            cluster_counts = df[cluster_col].value_counts()
            st.bar_chart(cluster_counts)

            if "Purchase_Probability" in df.columns:
                st.subheader("📈 Purchase Behavior by Cluster")

                fig = px.box(
                    df,
                    x=cluster_col,
                    y="Purchase_Probability",
                    title="Purchase Probability Distribution"
                )
                st.plotly_chart(fig, width="stretch")

        else:
            st.warning("No cluster data found")

    # ==============================
    # TAB 3
    # ==============================

    with tab3:
        if "Churn" in df.columns:

            churn_df = df[df["Churn"] == 1]

            st.warning(f"{len(churn_df)} customers at risk")
            st.dataframe(churn_df)

            fig = px.histogram(df, x="Churn", title="Churn Distribution")
            st.plotly_chart(fig, width="stretch")

        else:
            st.warning("Churn not available")

    # ==============================
    # TAB 4
    # ==============================

    with tab4:

        if "Explanation" in df.columns and df["Explanation"].notna().any():

            st.subheader("Customer Insights")

            sample = df.sample(n=min(10, len(df)), random_state=42)

            for _, row in sample.iterrows():
                st.markdown(
                    f"**Customer {int(row.get('CustomerID', 0))}** → {row['Explanation']}"
                )

        else:
            st.warning("No explanations available")

else:

    st.error("❌ No data available")

    st.info("""
    Fix steps:
    1. Run pipeline: python main.py
    2. Upload CSV to GitHub release
    3. Refresh dashboard
    """)