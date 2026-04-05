"""
Streamlit dashboard for Customer Segmentation AI System
Downloads real results from GitHub Release
"""
import streamlit as st
import pandas as pd
import os
import subprocess
from datetime import datetime

st.set_page_config(
    page_title="AI Customer Segmentation System",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 AI Customer Segmentation System")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(BASE_DIR, "outputs", "customer_segments.csv")
cache_dir = os.path.join(BASE_DIR, ".streamlit_cache")

@st.cache_resource
def download_from_release():
    """Download the latest results from GitHub Release"""
    try:
        st.info("📥 Downloading latest results from GitHub Release...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download from latest release
        # This URL fetches the latest release's assets
        download_url = "https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases/download/latest/customer_segments.csv"
        
        # Alternative: Use GitHub API to get latest release
        try:
            import urllib.request
            st.write("🔍 Checking for latest release...")
            
            # Get latest release info from GitHub API
            api_url = "https://api.github.com/repos/VenuOnTech/Customer-Segmentation-AI-System/releases/latest"
            with urllib.request.urlopen(api_url) as response:
                import json
                release_data = json.loads(response.read().decode())
                
                # Find the CSV file in assets
                if 'assets' in release_data and len(release_data['assets']) > 0:
                    csv_asset = next((asset for asset in release_data['assets'] 
                                    if asset['name'] == 'customer_segments.csv'), None)
                    
                    if csv_asset:
                        download_url = csv_asset['browser_download_url']
                        st.write(f"✅ Found release: {release_data['tag_name']}")
                    else:
                        st.warning("⚠️ CSV file not found in latest release")
                        return None
        except Exception as e:
            st.warning(f"⚠️ Could not fetch latest release info: {str(e)}")
            st.write("Attempting direct download...")
        
        # Download the file
        st.write(f"📥 Downloading from: {download_url}")
        result = subprocess.run(
            ["curl", "-L", "-o", output_path, download_url],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            st.success("✅ Data downloaded successfully!")
            return pd.read_csv(output_path)
        else:
            st.error(f"❌ Download failed: {result.stderr}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error downloading data: {str(e)}")
        return None

# Try to load cached data first
df = None

if os.path.exists(output_path):
    try:
        df = pd.read_csv(output_path)
        st.info(f"📊 Loaded cached results (Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    except Exception as e:
        st.warning(f"⚠️ Could not load cached file: {str(e)}")

# If no cached data, download from release
if df is None:
    with st.spinner("Fetching data..."):
        df = download_from_release()

# Display results if data loaded
if df is not None and len(df) > 0:
    
    st.success("✅ Segmentation Results Loaded Successfully!")
    
    st.divider()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Customers", f"{len(df):,}")
    with col2:
        st.metric("🎯 Number of Segments", int(df["Cluster"].nunique()) if "Cluster" in df.columns else "N/A")
    with col3:
        if "Churn" in df.columns:
            churn_count = (df["Churn"] == 1).sum()
            st.metric("⚠️ At-Risk Customers", f"{int(churn_count):,}")
        else:
            st.metric("⚠️ At-Risk Customers", "N/A")
    with col4:
        if "Purchase_Probability" in df.columns:
            avg_prob = df["Purchase_Probability"].mean()
            st.metric("📈 Avg Purchase Probability", f"{avg_prob:.2%}")
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Customer Data", "📈 Cluster Analysis", "⚠️ Churn Risk", "🔍 Explanations"])
    
    with tab1:
        st.subheader("Customer Segmentation Data")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Cluster Distribution")
        if "Cluster" in df.columns:
            cluster_counts = df["Cluster"].value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            # Cluster statistics
            st.write("**Cluster Statistics:**")
            for cluster_id in sorted(df["Cluster"].unique()):
                cluster_data = df[df["Cluster"] == cluster_id]
                st.write(f"- **Cluster {int(cluster_id)}**: {len(cluster_data)} customers")
        else:
            st.warning("Cluster column not found")
    
    with tab3:
        st.subheader("Churn Risk Analysis")
        if "Churn" in df.columns:
            churn_df = df[df["Churn"] == 1]
            
            if len(churn_df) > 0:
                st.warning(f"⚠️ {len(churn_df)} customers at risk of churn")
                st.dataframe(churn_df, use_container_width=True)
                
                # Download churn list
                csv = churn_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download At-Risk Customers",
                    data=csv,
                    file_name="at_risk_customers.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ No customers at risk of churn!")
        else:
            st.warning("Churn column not found")
    
    with tab4:
        st.subheader("Customer Explanations")
        if "Explanation" in df.columns:
            # Show sample explanations
            st.write("**Why customers are in their segments:**")
            
            sample_size = min(10, len(df))
            for idx in df.sample(n=sample_size).index:
                customer_id = df.loc[idx, "CustomerID"]
                explanation = df.loc[idx, "Explanation"]
                cluster = df.loc[idx, "Cluster"] if "Cluster" in df.columns else "N/A"
                
                st.markdown(f"""
                - **Customer {int(customer_id)}** (Cluster {int(cluster)}): {explanation}
                """)
        else:
            st.warning("Explanation column not found")

else:
    st.error("❌ Could not load segmentation results")
    
    st.warning("⚠️ **What to do:**")
    st.markdown("""
    1. **Check GitHub Actions**: Ensure the pipeline ran successfully
    2. **Check GitHub Releases**: Verify that `customer_segments.csv` is uploaded to a Release
    3. **Verify Release Tag**: The release should have a tag like `v1.0.0-outputs-X`
    
    **Manual Fix:**
    - Run the pipeline locally: `python main.py`
    - Create a GitHub Release and upload `outputs/customer_segments.csv`
    - Refresh this page
    """)
    
    st.info("📚 **For more info:**")
    st.markdown("""
    - [View Pipeline Status](https://github.com/VenuOnTech/Customer-Segmentation-AI-System/actions)
    - [View Releases](https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases)
    """)
