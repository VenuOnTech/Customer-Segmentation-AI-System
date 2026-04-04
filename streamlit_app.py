"""
Main entry point for Streamlit app
Streamlit Cloud will run this file
"""
import streamlit as st
import pandas as pd
import os
import sys

# Page configuration MUST come first
st.set_page_config(
    page_title="AI Customer Segmentation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
        .main { padding: 0rem 1rem; }
        h1 { color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application logic"""
    st.title("🎯 AI Customer Segmentation System")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio("Select Page", ["Home", "Segmentation Results", "Analytics"])
    
    # Home page
    if page == "Home":
        st.markdown("""
            ### Welcome to the Customer Segmentation AI System
            
            This application uses machine learning to segment customers based on their behavior patterns.
            
            **Features:**
            - 📊 RFM Analysis (Recency, Frequency, Monetary)
            - 🤖 K-Means Clustering
            - 🎯 Customer Churn Prediction
            - 📈 Future Purchase Forecasting
        """)
        
        # Check if data exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "outputs", "customer_segments.csv")
        
        if os.path.exists(output_path):
            st.success("✅ Segmentation results are available!")
        else:
            st.warning("⚠️ No segmentation results found. Run the pipeline first.")
            st.info("Execute: `python main.py` to generate results")
    
    # Segmentation Results page
    elif page == "Segmentation Results":
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "outputs", "customer_segments.csv")
        
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            
            st.subheader("📊 Customer Segments Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", len(df))
            with col2:
                if "Cluster" in df.columns:
                    st.metric("Number of Segments", df["Cluster"].nunique())
            with col3:
                if "Churn" in df.columns:
                    st.metric("At-Risk Customers", (df["Churn"] == 1).sum())
            
            st.divider()
            
            # Display dataframe
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
                st.dataframe(churn_df, use_container_width=True)
                st.info(f"Total at-risk customers: {len(churn_df)}")
        else:
            st.error("❌ Segmentation results not found!")
            st.info("Run the ML pipeline: `python main.py`")
    
    # Analytics page
    elif page == "Analytics":
        st.subheader("📊 Analytics Dashboard")
        st.info("Analytics features coming soon...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please check the logs for more details")
        import traceback
        st.write(traceback.format_exc())