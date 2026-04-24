# 🎯 AI Customer Segmentation & Predictive Analytics System

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-red?logo=streamlit)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Enabled-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-green?logo=github)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**An autonomous, enterprise-grade ML pipeline combining Reinforcement Learning, Deep Neural Networks, and Explainable AI (SHAP) for real-time customer segmentation and churn prediction.**

[Live Dashboard](https://customer-segmentation-ai-system-xqev9fmhnpgwe6vp2kgz2b.streamlit.app/) • [GitHub Issues](https://github.com/VenuOnTech/Customer-Segmentation-AI-System/issues) • [Releases](https://github.com/VenuOnTech/Customer-Segmentation-AI-System/releases)

</div>

---

## 📚 Table of Contents

- [Overview](#-overview)
- [System Architecture](#️-system-architecture)
- [Key AI Capabilities](#-key-ai-capabilities)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Data Dictionary (Output)](#-data-dictionary-output)
- [MLOps & Monitoring](#mlops)
- [License & Support](#-license--support)

---

## 🎯 Overview

Traditional customer segmentation relies on static, rule-based groupings. This project introduces a **Hybrid AI Architecture** that dynamically processes data streams, extracts complex behavioral patterns using Deep Learning (Autoencoders & LSTMs), optimizes clustering via Reinforcement Learning, and translates black-box predictions into human-readable insights using Explainable AI (SHAP).

Designed with robust MLOps principles, the system features automated data validation, statistical drift monitoring, and an end-to-end CI/CD pipeline powered by GitHub Actions.

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION & STREAMING LAYER                                    │
│ Simulates real-time data streaming • Validates schema • Cleans nulls   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ 2. DYNAMIC FEATURE ENGINEERING LAYER                                   │
│ • Baseline: RFM (Recency, Frequency, Monetary)                         │
│ • Behavioral & Temporal: Purchase velocity, variances, intervals       │
│ • Deep Extraction: Neural Network Autoencoder latent features          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ 3. HYBRID AI DECISION ENGINE                                           │
│ ├── Clustering: K-Means optimized autonomously by Q-Learning (RL) agent│
│ ├── Sequential Prediction: LSTM networks predicting behavioral flow    │
│ └── Classification: Deep Multi-Layer Perceptron (MLP) for Churn Risk   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ 4. MLOPS & EXPLAINABILITY LAYER                                        │
│ ├── XAI: SHAP values translated into plain-text business logic         │
│ ├── Monitoring: Kolmogorov-Smirnov tests for behavioral data drift     │
│ └── Tracking: Data lineage JSON and automated Experiment logging       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
┌───────────▼───────────┐                     ┌───────────▼───────────┐
│ Streamlit UI          │                     │ FastAPI Backend       │
│ Interactive Dashboard │                     │ REST Endpoints for    │
│ (Visual Analytics)    │                     │ external integrations │
└───────────────────────┘                     └───────────────────────┘
```

## 🧠 Key AI Capabilities

### 1. Reinforcement Learning Optimized Clustering
Instead of hardcoding the number of customer segments, the system utilizes an RL agent (`rl_optimizer.py`) that explores different values of $k$ and maximizes the silhouette score to automatically find the optimal cluster boundaries.

### 2. Deep Predictive Modeling (MLP & LSTM)
- **Churn Prediction:** Uses a Multi-Layer Perceptron (`deep_churn_model.py`) to classify high-risk customers, autonomously generating business-logic labels if raw data is unlabelled.
- **Behavioral Scoring:** An LSTM network (`lstm_churn_model.py`) processes the historical sequence of customer transactions to score future engagement trajectory.

### 3. Explainable AI (SHAP)
Black-box models are converted into transparent business rules. Using SHAP (`shap_explainer.py`), the system outputs exact textual reasons for its decisions (e.g., "Churn risk influenced by high recency & low frequency").

### 4. Continuous Data Drift Monitoring
The pipeline automatically runs two-sample Kolmogorov-Smirnov tests (`behavior_drift.py`) against historical feature stores to detect statistical shifts in customer purchasing frequencies, alerting the system when retraining is necessary.

---

## 📦 Quick Start

### Prerequisites
- Python 3.10+
- 4GB RAM minimum (TensorFlow/PyTorch models gracefully degrade if memory is constrained).

### Installation

```bash
# 1. Clone the repository
git clone [https://github.com/VenuOnTech/Customer-Segmentation-AI-System.git](https://github.com/VenuOnTech/Customer-Segmentation-AI-System.git)  
cd Customer-Segmentation-AI-System

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 💻 Usage Guide

### 1. Run the Autonomous ML Pipeline
Executes the full end-to-end architecture (streaming, engineering, clustering, XAI, monitoring).

```bash
python main.py
```
Outputs generated in `outputs/` and versioned models in `models/v*/`.

### 2. Launch the Streamlit Dashboard
Visualize the live segmentation results, view SHAP insights, and analyze churn metrics.

```bash
streamlit run app/dashboard.py
```
Access at: http://localhost:8501  

### 3. Start the FastAPI Serving Backend
Expose the trained models as REST API endpoints for real-time integration.

```bash
uvicorn api.app:app --reload
```
Access API Docs at: http://localhost:8000/docs

4. Run the CI/CD Test Suite
```bash
python tests/test_pipeline.py
```

## 📁 Project Structure
```plaintext
Customer-Segmentation-AI-System/
├── 🎯 main.py                  # Core pipeline orchestrator
├── 🤖 api/app.py               # FastAPI serving layer
├── 📊 app/dashboard.py         # Streamlit UI
├── ⚙️ config/                  # YAML system configs & schema aliases
├── 📈 data/                    # Raw & processed datasets
├── 📦 models/                  # Versioned serialized models (.pkl) & metadata
├── 📤 outputs/                 # CSV outputs, feature stores, JSON logs
├── 🧪 tests/                   # Unit test suite
└── 🔧 src/                     # Core ML Modules
    ├── data_ingestion/         # Versioning, loaders, strict schema detection
    ├── preprocessing/          # automated cleaning and validation
    ├── feature_engineering/    # RFM, temporal, autoencoders, behavioral
    ├── feature_store/          # Feature persistence
    ├── segmentation/           # RL-optimized K-Means, DBSCAN
    ├── prediction/             # Deep MLP, LSTM, mathematical probability
    ├── explainability/         # Global/Local SHAP explanations
    ├── monitoring/             # KS drift detection, data lineage, quality reports
    ├── optimization/           # Pipeline and RL optimizers
    ├── streaming/              # Data batch stream simulator
    └── utils/                  # Experiment tracking and config loaders
```

## 📖 Data Dictionary (Output)
When viewing outputs/customer_segments.csv or the dashboard, here is how to interpret the engineered metrics:

| Feature | Description | Business Interpretation |
| :--- | :--- | :--- |
| **Recency** | Days since last purchase | High = Churn Risk. Low = Active. |
| **Frequency** | Total number of purchases | High = Loyal customer. |
| **Monetary** | Total lifetime spend | High = VIP/Whale. |
| **Avg_Interval** | Average days between visits | Measures routine. Low = Frequent shopper. |
| **Std_Quantity** | Variance in basket size | High = Erratic buyer. Low = Predictable routine. |
| **Purchase_Velocity**| Purchases / Customer Lifetime | Speed of engagement. |
| **LSTM_Score** | Sequential neural network output | High = Sequence predicts strong future engagement. |
| **Purchase_Prob.** | Algorithmic likelihood of return | 0.0 to 1.0 scoring metric. |
| **Churn** | Deep Learning risk classification | 1 = At Risk. 0 = Safe. |
| **Explanation** | Explainable AI (SHAP) text | Plain English reasoning for the model's decision. |

## <a id="mlops"></a>🛡️ MLOps & Monitoring

This project utilizes advanced MLOps practices:

* **Automated CI/CD:** Push to `main` triggers GitHub Actions (`.github/workflows/main.yml`) which runs tests, executes the pipeline, and auto-publishes the models/data to GitHub Releases.
* **Graceful Degradation:** The pipeline automatically detects environment constraints (e.g., missing TensorFlow in CI) and gracefully falls back to core features without crashing.
* **Data Lineage:** Every run logs `outputs/data_lineage.json` linking the exact data hash to the exact output file.
* **Experiment Tracking:** Hyperparameters, silhouette scores, and neural network accuracies are automatically logged to `outputs/experiments.json`.

---

## 📄 License & Support

This project is licensed under the MIT License - see the LICENSE file for details.

* **Issues & Questions:** Please use the GitHub Issues tab.
* **Contact:** venudadi11@gmail.com

⭐ Star this repo if you find it useful!

Made with ❤️ by VenuOnTech

🔗 GitHub • 📊 Dashboard • 📧 Contact

</div>
