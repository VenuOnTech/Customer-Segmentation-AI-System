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


# 🔥 SAFE LOCAL LOAD
@st.cache_data
def load_local_data():
    try:
        if os.path.exists(output_path):
            return pd.read_csv(output_path)
    except Exception as e:
        st.warning(f"⚠️ Error loading local data: {e}")
    return None


# 🔥 SAFE DOWNLOAD
@st.cache_data
def download_from_release():
    try:
        st.info("📥 Fetching latest data from GitHub Release...")

        url = "https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases/latest/download/customer_segments.csv"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        result = subprocess.run(
            ["curl", "-L", "-o", output_path, url],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and os.path.exists(output_path):
            return pd.read_csv(output_path)

        st.error("❌ Failed to download data")
        return None

    except Exception as e:
        st.error(f"❌ Download error: {e}")
        return None


# 🔥 LOAD FLOW
df = load_local_data()

if df is None:
    with st.spinner("Fetching data..."):
        df = download_from_release()


# 🔥 UI DISPLAY
if df is not None and len(df) > 0:

    st.success("✅ Segmentation Results Loaded Successfully!")

    st.divider()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Customers", f"{len(df):,}")

    with col2:
        st.metric("🎯 Segments", df["Cluster"].nunique() if "Cluster" in df else "N/A")

    with col3:
        if "Churn" in df:
            st.metric("⚠️ At-Risk", int((df["Churn"] == 1).sum()))
        else:
            st.metric("⚠️ At-Risk", "N/A")

    with col4:
        if "Purchase_Probability" in df:
            st.metric("📈 Avg Probability", f"{df['Purchase_Probability'].mean():.2%}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Data", "📈 Clusters", "⚠️ Churn", "🔍 Insights"]
    )

    with tab1:
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "data.csv")

    with tab2:
        if "Cluster" in df:
            st.bar_chart(df["Cluster"].value_counts().sort_index())

    with tab3:
        if "Churn" in df:
            churn_df = df[df["Churn"] == 1]
            st.dataframe(churn_df)

    with tab4:
        if "Explanation" in df:
            sample = df.sample(min(10, len(df)))
            for _, row in sample.iterrows():
                st.write(
                    f"Customer {int(row['CustomerID'])} → {row['Explanation']}"
                )

else:
    st.error("❌ No data available")

    st.markdown("""
    ### Fix Steps:
    1. Run pipeline (`python main.py`)
    2. Check GitHub Actions
    3. Verify release contains CSV
    """)