"""
Streamlit dashboard for Customer Segmentation AI System
Stable + Production-safe version (NO curl dependency)
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import urllib.request
import json

st.set_page_config(
    page_title="AI Customer Segmentation System",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 AI Customer Segmentation System")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(BASE_DIR, "outputs", "customer_segments.csv")


@st.cache_data
def download_from_release():
    """Download latest CSV from GitHub Release safely"""
    
    try:
        st.info("📥 Fetching latest release data...")

        api_url = "https://api.github.com/repos/VenuOnTech/Customer-Segmentation-AI-System/releases/latest"

        with urllib.request.urlopen(api_url, timeout=10) as response:
            release_data = json.loads(response.read().decode())

        if "assets" not in release_data:
            st.error("❌ No assets found in release")
            return None

        csv_asset = next(
            (a for a in release_data["assets"] if a["name"] == "customer_segments.csv"),
            None
        )

        if not csv_asset:
            st.error("❌ customer_segments.csv not found in release")
            return None

        download_url = csv_asset["browser_download_url"]

        # Download file safely
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        urllib.request.urlretrieve(download_url, output_path)

        st.success(f"✅ Loaded from release: {release_data['tag_name']}")

        return pd.read_csv(output_path)

    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        return None


# 🔹 LOAD DATA
df = None

if os.path.exists(output_path):
    try:
        df = pd.read_csv(output_path)
        st.info(f"📊 Loaded cached data ({datetime.now().strftime('%H:%M')})")
    except:
        df = None

if df is None:
    df = download_from_release()


# 🔹 DISPLAY
if df is not None and not df.empty:

    st.success("✅ Segmentation Results Loaded")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Customers", f"{len(df):,}")

    with col2:
        st.metric("🎯 Segments",
                  int(df["Cluster"].nunique()) if "Cluster" in df else "N/A")

    with col3:
        if "Churn" in df:
            st.metric("⚠️ Churn",
                      int((df["Churn"] == 1).sum()))
        else:
            st.metric("⚠️ Churn", "N/A")

    with col4:
        if "Purchase_Probability" in df:
            st.metric("📈 Avg Prob",
                      f"{df['Purchase_Probability'].mean():.2%}")
        else:
            st.metric("📈 Avg Prob", "N/A")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data", "📈 Clusters", "⚠️ Churn", "🔍 Explain"
    ])

    with tab1:
        st.dataframe(df, use_container_width=True)

    with tab2:
        if "Cluster" in df:
            st.bar_chart(df["Cluster"].value_counts())
        else:
            st.warning("No Cluster column")

    with tab3:
        if "Churn" in df:
            churn_df = df[df["Churn"] == 1]
            st.write(f"At-risk customers: {len(churn_df)}")
            st.dataframe(churn_df)
        else:
            st.warning("No Churn column")

    with tab4:
        if "Explanation" in df:
            sample = df.sample(min(5, len(df)))
            for _, row in sample.iterrows():
                st.write(
                    f"Customer {row.get('CustomerID','?')} → {row['Explanation']}"
                )
        else:
            st.warning("No Explanation column")

else:
    st.error("❌ Failed to load data")

    st.markdown("""
### 🔧 Fix Checklist:
- Ensure GitHub Actions ran successfully
- Ensure Release contains `customer_segments.csv`
- Try running locally: `python main.py`
""")
