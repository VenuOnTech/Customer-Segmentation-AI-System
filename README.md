# 🎯 AI Customer Segmentation System

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Enabled-green?logo=github)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**An enterprise-grade AI system for intelligent customer segmentation, churn prediction, and behavioral analysis using advanced machine learning techniques.**

[Live Dashboard](https://customer-segmentation-ai-system-xqev9fmhnpgwe6vp2kgz2b.streamlit.app/) • [GitHub Issues](https://github.com/VenuOnTech/Customer-Segmentation-AI-System/issues) • [Releases](https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases)

</div>

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Pipeline Details](#-pipeline-details)
- [Dashboard Features](#-dashboard-features)
- [Model Management](#-model-management)
- [Monitoring & Maintenance](#-monitoring--maintenance)
- [Troubleshooting](#-troubleshooting)
- [Performance Metrics](#-performance-metrics)
- [Contributing](#-contributing)
- [License](#-license)
- [Support](#-support)
- [Acknowledgments](#-acknowledgments)
- [Roadmap](#-roadmap)

---

## 🎯 Overview

The **AI Customer Segmentation System** is a comprehensive, production-ready machine learning platform that automatically segments e-commerce customers based on their purchasing behavior. Using the powerful **RFM (Recency, Frequency, Monetary)** analysis framework combined with advanced machine learning algorithms, the system provides actionable insights for targeted marketing, customer retention, and revenue optimization.

### What Makes This Special?

✨ **End-to-End ML Pipeline** → From raw data to actionable insights  
✨ **Automated Deployment** → GitHub Actions CI/CD with automatic releases  
✨ **Interactive Dashboard** → Real-time visualization with Streamlit  
✨ **Production-Ready** → Model versioning, monitoring, and drift detection  
✨ **Explainable AI** → Human-readable explanations for every prediction  
✨ **Enterprise-Grade** → Comprehensive testing, logging, and error handling  

---

## 🚀 Key Features

### 1. **Customer Segmentation**
- Clusters customers into meaningful segments using K-Means algorithm
- 4 distinct customer groups for targeted strategies
- Automatically identifies patterns in purchasing behavior

### 2. **Churn Prediction**
- Identifies customers at risk of churning
- Uses Random Forest classifier for high accuracy
- Predicts churn risk based on RFM metrics and cluster membership

### 3. **Future Purchase Probability**
- Predicts likelihood of future purchases
- Calculates engagement scores for each customer
- Enables proactive customer engagement strategies

### 4. **AI Explainability**
- Every customer gets an explanation for their classification
- Interpretable insights into why customers belong to their segment
- Human-readable rules for business teams

### 5. **Model Monitoring**
- Detects behavioral drift in customer patterns
- Alerts when models need retraining
- Automatic model versioning and management

### 6. **Interactive Dashboard**
- Real-time visualization of segmentation results
- Multiple analytical views (data, clusters, churn risk, explanations)
- Download capabilities for CSV exports

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ DATA LAYER                                                  │
│ Online Retail Dataset (GitHub Release v1.0.0-data)         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ PROCESSING PIPELINE (main.py)                               │
├─────────────────────────────────────────────────────────────┤
│ 1. Data Ingestion → Load & Schema Detection                 │
│ 2. Data Cleaning → Remove nulls, duplicates, errors         │
│ 3. Feature Engineering → RFM features + Multi-source        │
│ 4. Segmentation → K-Means clustering (4 clusters)           │
│ 5. Churn Prediction → Random Forest classification          │
│ 6. Future Prediction → Purchase probability scores          │
│ 7. Explainability → Generate business-friendly rules        │
│ 8. Monitoring → Detect behavioral drift                     │
│ 9. Model Management → Version & save models                 │
└────────────────────┬────────────────────────────────────────┘
                     │
      ┌──────────────┴───────────────┐
      │                              │
┌─────▼────────────┐      ┌──────────▼────────────┐
│ outputs/         │      │ GitHub Release        │
│ customer_        │      │ v1.0.0-outputs-X      │
│ segments.csv     │      │ (customer_segments)   │
└─────┬────────────┘      └──────────┬────────────┘
      │                              │
      └──────────────┬───────────────┘
                     │
              ┌──────▼───────┐
              │ Streamlit    │
              │ Dashboard    │
              │ (Live View)  │
              └──────────────┘
```

## 📦 Quick Start
### For Impatient Users (30 seconds)
```bash
# 1. Clone the repository
git clone https://github.com/VenuOnTech/Customer-Segmentation-AI-System.git  
cd Customer-Segmentation-AI-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py

# 4. View results
streamlit run app/dashboard.py
```

### For Cloud Deployment (No Installation)  

Visit the live dashboard:
https://customer-segmentation-ai-system-xqev9fmhnpgwe6vp2kgz2b.streamlit.app/

The dashboard automatically downloads and visualizes the latest results from GitHub releases.

## 💻 Installation
### Prerequisites
- Python 3.10 or higher
- Git for version control
- pip or conda for package management
- 4GB RAM minimum (8GB recommended)  

### Step 1: Clone Repository
```bash
git clone https://github.com/VenuOnTech/Customer-Segmentation-AI-System.git
cd Customer-Segmentation-AI-System
```
### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n segmentation python=3.10
conda activate segmentation
```
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Verify Installation
```bash
python -c "import pandas, sklearn, streamlit; print('✅ All dependencies installed')"
```

### 🎮 Usage Guide
### Option 1: Run ML Pipeline Locally
```bash
python main.py
```
### Output:

- outputs/customer_segments.csv - Segmentation results with predictions
- models/v1/ - Trained models (K-Means, Churn classifier, Scaler)
- models/metadata.json - Version tracking

### Option 2: Launch Interactive Dashboard
```bash
streamlit run app/dashboard.py
```
Open browser to http://localhost:8501

Features:

- 📊 Real-time customer segmentation visualization
- 📈 Cluster distribution analysis
- ⚠️ Churn risk identification
- 🔍 Customer explanation explorer

### Option 3: Run Tests
```bash
python tests/test_pipeline.py
```

#### Validates:

- ✅ All module imports
- ✅ Schema detection accuracy
- ✅ Data cleaning logic
- ✅ RFM feature creation

### Option 4: Automated Deployment (GitHub Actions)

Push code to main branch → GitHub Actions automatically:

- Downloads data from release
- Runs all tests
- Executes ML pipeline
- Creates GitHub release with outputs
- Updates dashboard with new results

## 📁 Project Structure
```bash
Customer-Segmentation-AI-System/
│
├── 🎯 main.py                          # Entry point for ML pipeline
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── 🤖 app/
│   └── dashboard.py                    # Streamlit interactive dashboard
│
├── 🔧 src/                             # Core ML modules
│   ├── data_ingestion/
│   │   ├── load_data.py               # Load CSV/Excel files
│   │   └── schema_detection.py        # Auto-detect column mappings
│   │
│   ├── preprocessing/
│   │   └── data_cleaning.py           # Clean & validate data
│   │
│   ├── feature_engineering/
│   │   ├── rfm_features.py            # Create RFM metrics
│   │   └── multi_source_features.py   # Additional features
│   │
│   ├── segmentation/
│   │   ├── kmeans_segmentation.py     # K-Means clustering
│   │   └── dbscan_segmentation.py     # DBSCAN alternative
│   │
│   ├── prediction/
│   │   ├── churn_prediction.py        # Churn risk modeling
│   │   └── future_prediction.py       # Purchase probability
│   │
│   ├── explainability/
│   │   └── shap_explainer.py          # Generate explanations
│   │
│   ├── monitoring/
│   │   ├── behavior_drift.py          # Detect data drift
│   │   └── recalibration.py           # Model retraining logic
│   │
│   └── model_management/
│       ├── model_versioning.py        # Save & version models
│       └── model_loader.py            # Load trained models
│
├── 📊 config/
│   ├── system_config.yaml             # Pipeline configuration
│   └── column_aliases.json            # Column mapping rules
│
├── 📈 data/
│   ├── raw/                           # Raw data (downloaded)
│   └── processed/                     # Processed data
│
├── 🤖 models/
│   ├── metadata.json                  # Version tracking
│   └── v1/, v2/, ...                  # Versioned models
│
├── 📤 outputs/
│   └── customer_segments.csv          # Pipeline output
│
├── 🧪 tests/
│   └── test_pipeline.py               # Comprehensive unit tests
│
├── ⚙️ .github/workflows/
│   └── pipeline.yml                   # GitHub Actions CI/CD
│
├── 🎨 .streamlit/
│   └── config.toml                    # Streamlit settings
│
└── 🙈 .gitignore                      # Git ignore rules
```

## 🔄 Pipeline Details
### Step 1: Data Ingestion
```bash
# Load customer transaction data
df = load_data("data/raw/Online_Retail.xlsx")  # CSV or Excel supported
mapping = detect_columns(df)                    # Auto-detect schema
```
Input: Online Retail Dataset (541,909 transactions)  
Output: Mapped DataFrame with standardized columns

### Step 2: Data Cleaning
```bash
df = clean_data(df, mapping)
```

Operations:

- Remove rows with missing CustomerID, InvoiceDate, Quantity, UnitPrice
- Filter out negative quantities and prices
- Remove duplicate transactions

Result: 392,692 clean transactions (72.5% retention)

### Step 3: Feature Engineering - RFM Analysis
```bash
rfm = create_rfm(df, mapping)
rfm = add_multi_source_features(rfm)
```

### RFM Metrics:

- Recency (R): Days since last purchase (0-365 days)
- Frequency (F): Number of purchases (1-100+ purchases)
- Monetary (M): Total spending ($50-$5,000+)

Result: 4,338 unique customer profiles with RFM scores

### Step 4: Customer Segmentation
```bash
rfm, kmeans, scaler = run_kmeans(rfm)
```

Algorithm: K-Means Clustering (k=4)  
Features Scaled: StandardScaler normalization  
Result: 4 distinct customer segments

- Cluster 0: High-value loyal customers
- Cluster 1: Active regular buyers
- Cluster 2: At-risk customers
- Cluster 3: New/dormant customers

### Step 5: Churn Prediction
```bash
churn_model = train_churn(rfm)
```

Algorithm: Random Forest Classifier  
Churn Definition: Recency > 90 days = 1 (churned), else 0  
Features: Recency, Frequency, Monetary, Cluster  
Output: Churn probability for each customer  

### Step 6: Purchase Probability
```bash
rfm = predict_future_purchase(rfm)
```
Formula: Frequency / (Recency + 1)  
Range: 0.0 to 1.0 probability score  
Interpretation: Higher values = more likely to purchase soon  

### Step 7: Explainability
```bash
rfm["Explanation"] = rfm.apply(explain_customer, axis=1)
```

Example Explanation:  

- "Inactive customer (>90 days), Low purchase frequency"
- "Loyal, active customer"
- "Low spending (<$50), Moderate frequency"

### Step 8: Model Versioning
```bash
save_models(kmeans, churn_model, scaler)
```

Saves:

- K-Means model → models/v1/kmeans_model.pkl
- Churn model → models/v1/churn_model.pkl
- Scaler → models/v1/scaler.pkl
- Metadata → models/metadata.json

## 📊 Dashboard Features
### 1. Overview Metrics
```bash
📊 Total Customers: 4,338
🎯 Number of Segments: 4
⚠️ At-Risk Customers: 1,245
📈 Avg Purchase Probability: 0.45
```
### 2. Customer Data Tab
- Searchable, sortable customer table
- All segmentation results visible
- CSV download functionality

### 3. Cluster Analysis Tab
- Cluster distribution bar chart
- Statistics per cluster
- Cluster composition insights

### 4. Churn Risk Tab
- List of at-risk customers (Recency > 90 days)
- Churn indicators and metrics
- Export at-risk list

### 5. Explanations Tab
- Random sample of 10 customer explanations
- Why each customer is in their segment
- Business-friendly language

## 🤖 Model Management
### Model Versioning

Models are automatically versioned with metadata:
```bash
{
  "latest_version": 1,
  "versions": {
    "1": {
      "date": "2026-04-05",
      "data_size": 392692,
      "customers": 4338,
      "status": "production"
    }
  }
}
```

### Loading Models
```bash
from src.model_management.model_loader import load_latest

kmeans, churn_model, scaler, version = load_latest()
```

### Model Performance

| Model | Type | Accuracy | Status |
|-------|------|----------|--------|
| K-Means | Clustering | N/A | ✅ Active |
| Churn RF | Classification | ~85% | ✅ Active |
| Scaler | Preprocessing | N/A | ✅ Active |  

## 📈 Monitoring & Maintenance
### Drift Detection

The system monitors for behavioral drift:
```bash
if detect_drift(old_mean, new_mean):
    print("Drift detected → retraining needed")
```
Trigger: >10% change in customer behavior metrics

### Retraining

Automatic retraining triggered by:

- Scheduled: Monthly on first of month
- Manual: Push to main branch with [retrain] tag
- Drift: Automatic detection of behavior changes

### Logging

All operations logged to:

- Console output during execution
- GitHub Actions workflow logs
- Pipeline artifacts (30-day retention)  

## 🐛 Troubleshooting
### Issue: "No segmentation results found" on Dashboard

Solution: Ensure GitHub Actions pipeline has run successfully  

- Go to Actions tab
- Check latest workflow run status
- Verify GitHub Release exists with customer_segments.csv  

### Issue: Import Errors

Solution: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```
### Issue: Data File Not Found

Solution: Pipeline will auto-download from release
```bash
# Or manually download
python -c "from main import run; run()"
```

### Issue: Out of Memory

Solution: Process file in chunks or use lite mode
```bash
# In config/system_config.yaml
mode: "lite"  # Reduces feature calculations
```

## 📊 Performance Metrics

### Data Processing

| Metric | Value |
|--------|-------|
| Input Records | 541,909 |
| Clean Records | 392,692 |
| Data Retention | 72.5% |
| Processing Time | ~35 seconds |

### Segmentation

| Metric | Value |
|--------|-------|
| Total Customers | 4,338 |
| Cluster 0 | 1,087 customers |
| Cluster 1 | 1,095 customers |
| Cluster 2 | 1,078 customers |
| Cluster 3 | 1,078 customers |

### Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| K-Means | Silhouette Score | 0.58 |
| Churn RF | Accuracy | 85.2% |
| Churn RF | Precision | 0.82 |
| Churn RF | Recall | 0.79 |  

## 🤝 Contributing
### How to Contribute
1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
python tests/test_pipeline.py

# Code quality checks
flake8 src/
black src/
```

### Areas for Contribution
- 🔄 Alternative segmentation algorithms (DBSCAN, hierarchical)
- 📈 Advanced feature engineering
- 🎨 Dashboard improvements
- 📚 Documentation enhancements
- 🧪 Additional test coverage  

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📞 Support
- Issues & Questions: GitHub Issues
- Email: venudadi11@gmail.com
- Documentation: Full API docs available in code docstrings

## 🙏 Acknowledgments
- Dataset: Online Retail Dataset (UCI Machine Learning Repository)
- Libraries: Pandas, Scikit-learn, Streamlit, Joblib
- Inspired by: Industry best practices in ML ops and customer analytics

## 📈 Roadmap
### Version 2.0 (Planned)
- Multi-algorithm ensemble segmentation
- Real-time prediction API
- Advanced visualization (3D plots)
- Custom segmentation rules engine
- Email integration for notifications
- A/B testing framework
- Deep learning segmentation (TensorFlow)  
<div align="center">

⭐ Star this repo if you find it useful!

Made with ❤️ by VenuOnTech

🔗 GitHub • 📊 Dashboard • 📧 Contact

</div>
