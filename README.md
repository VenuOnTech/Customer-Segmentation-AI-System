# Customer Segmentation AI System - Pipeline Summary

## What This System Does
This system automatically segments customers based on their purchasing behavior using RFM analysis and machine learning.

## Pipeline Flow

1. **Data Loading** → Reads customer transaction data
2. **Schema Detection** → Automatically maps columns
3. **Data Cleaning** → Removes invalid data
4. **Feature Engineering** → Creates RFM features
5. **Segmentation** → K-Means clustering
6. **Churn Prediction** → Identifies at-risk customers
7. **Explainability** → Explains customer classifications
8. **Model Versioning** → Saves trained models

## Running the Pipeline

### Option 1: Local Execution
```bash
pip install -r requirements.txt
python main.py