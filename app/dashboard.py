import streamlit as st
import pandas as pd
import os
import requests
import io   # ✅ FIXED
import plotly.express as px

st.set_page_config(
    page_title="AI Customer Segmentation System",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 AI Customer Segmentation System")
st.markdown("### 🚀 Autonomous ML Pipeline with Explainable AI + Drift Monitoring")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_path = os.path.join(BASE_DIR, "outputs", "customer_segments.csv")


# ==============================
# LOAD DATA (SMART LOADER 🔥)
# ==============================

def load_data():
    url = "https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases/download/latest/customer_segments.csv"

    try:
        st.info("📥 Fetching latest data from GitHub Release...")
        response = requests.get(url)

        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            st.success("✅ Loaded latest release data")
            return df

    except Exception as e:
        st.warning(f"⚠️ GitHub fetch failed: {str(e)}")

    # ✅ FALLBACK
    if os.path.exists(local_path):
        st.info("📂 Loading local file...")
        return pd.read_csv(local_path)

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
    col3.metric(
        "⚠️ At Risk",
        int(
            ((df["Churn"] == 1) & (df["Purchase_Probability"] < 0.4)).sum()
        ) if "Churn" in df else "N/A"
    )
    col4.metric("📈 Avg Purchase", f"{df['Purchase_Probability'].mean():.2%}" if "Purchase_Probability" in df else "N/A")
    col5.metric("⏱️ Avg Recency", f"{df['Recency'].mean():.1f}" if "Recency" in df else "N/A")

    st.divider()

    if cluster_col and {"Frequency", "Monetary"}.issubset(df.columns):

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
        st.dataframe(df, use_container_width=True)

    with tab2:
        if cluster_col:
            st.bar_chart(df[cluster_col].value_counts())

    with tab3:
        if "Churn" in df:
            churn_df = df[
                (df["Churn"] == 1) &
                (df["Recency"] > df["Recency"].median())
            ]

            if len(churn_df) == 0:
                st.warning("⚠️ No churn customers detected")
            else:
                st.dataframe(churn_df, use_container_width=True)

    with tab4:
        st.subheader("🔍 Customer Insights")

        if "Explanation" in df.columns:

            df["Explanation"] = df["Explanation"].fillna("").astype(str).str.strip()

            invalid_values = ["", "Not computed", "Model explanation unavailable"]

            valid_df = df[~df["Explanation"].isin(invalid_values)]

            if len(valid_df) == 0:
                st.warning("⚠️ No valid insights available")
            else:
                sample = valid_df.sample(n=min(10, len(valid_df)))

                for _, row in sample.iterrows():
                    customer_id = row.get("CustomerID", "Unknown")
                    st.success(f"Customer {customer_id} → {row['Explanation']}")

        else:
            st.error("❌ Explanation column missing")

else:

    st.error("❌ No data available")