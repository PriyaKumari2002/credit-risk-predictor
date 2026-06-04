# 🏦 Credit Risk Prediction System
End-to-end ML pipeline for loan default prediction in Indian BFSI sector

**Python** • **XGBoost** • **FastAPI** • **MLflow** • **Streamlit** • **SHAP**

---

## 🎯 Business Problem

India's NBFC and fintech sector evaluates millions of thin-file borrowers with limited credit history every year. Traditional scoring models fail for first-time borrowers.

**The Cost of Getting It Wrong:**

| Mistake | What Happened | Cost |
|---------|---------------|------|
| False Negative | Approved a defaulter | ₹50,000 (principal lost) |
| False Positive | Rejected a good customer | ₹5,000 (interest lost) |

A false negative is **10x more expensive** than a false positive. This system is built around that reality.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Home Credit Default Risk (Kaggle) |
| Rows | 307,511 loan applications |
| Features | 122 raw → 193 after engineering |
| Default Rate | 8.1% (severe class imbalance) |
| Challenge | 40%+ missing values, outliers, imbalanced classes |

---

## 🏗️ End-to-End Architecture

```
Raw Data (307K rows, 122 features)
         ↓
EDA & Data Cleaning
- 41 high-missing columns dropped (>50% missing)
- Outliers capped at 99th percentile
- Class imbalance analysis and visualization
         ↓
Feature Engineering
- AGE_YEARS from DAYS_BIRTH
- YEARS_EMPLOYED from DAYS_EMPLOYED
- DEBT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
- CREDIT_TERM = AMT_CREDIT / AMT_ANNUITY
- One-hot encoding for 10 categorical columns
         ↓
Model Training Pipeline
- Baseline: Logistic Regression
- SMOTE applied ONLY on training data (no leakage)
- XGBoost vs LightGBM comparison
- Threshold optimization based on business cost matrix
         ↓
MLflow Experiment Tracking
- 4 experiments tracked with parameter/metric logging
- Model artifacts and metrics persisted
         ↓
SHAP Explainability
- Global feature importance analysis
- Per-prediction force plots for model interpretation
         ↓
Streamlit Dashboard (Deployed)
- Real-time credit analyst UI with model predictions
- SHAP visualizations and feature importance charts
```

---

## 📈 Model Comparison

| Model | ROC-AUC | Recall (Default) | Precision (Default) | Threshold |
|-------|---------|-----------------|-------------------|-----------|
| Logistic Regression | 0.63 | 0.58 | 0.12 | 0.50 |
| LightGBM | 0.73 | 0.33 | 0.22 | 0.20 |
| **XGBoost ✅** | **0.73** | **0.68** | **0.15** | **0.10** |

### Why XGBoost at threshold 0.1?

- Recall of 0.68 means **68% of real defaulters are caught**
- Threshold chosen based on **business cost matrix** — not default 0.5
- Missing a defaulter costs 10x more than rejecting a good customer
- **Result:** 15% improvement in defaulter detection vs baseline logistic regression

---

## 🔬 MLflow Experiment Tracking

All 4 experiments tracked with:
- Full parameter logging (model type, hyperparameters, feature counts)
- Metric comparison (ROC-AUC, Recall, Precision, F1)
- Model artifacts persisted for reproducibility

---

## 🧠 SHAP Explainability

**Global Feature Importance:** Identifies top features driving default predictions across the entire dataset

**Per-Prediction Force Plots:** Shows feature contributions for individual loan decisions, enabling credit officers to understand why a specific applicant was approved/rejected

---

## 🚀 Live Deployment & How to Run

### Live Streamlit Dashboard
👉 **[View Live App](https://credit-risk-predictor-yzpt7s57mfzkbnhmajqfmx.streamlit.app/)**

Fully functional dashboard with:
- Real-time model inference
- SHAP feature importance visualizations
- Interactive prediction interface
- Model comparison charts

### Run Locally

#### Prerequisites
```bash
git clone https://github.com/PriyaKumari2002/credit-risk-predictor
cd credit-risk-predictor
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

#### View MLflow UI (if running locally)
```bash
python -m mlflow ui
# Access at http://localhost:5000
```

#### FastAPI Endpoint (Local Development)
The FastAPI endpoint is fully implemented in `api/main.py` and can be run locally:
```bash
python -m uvicorn api.main:app --reload
# Access at http://localhost:8000/docs for interactive API docs
```

**Note:** Streamlit Cloud deployment is production-ready. FastAPI endpoint designed for local development and can be containerized for production deployment.

---

## 📁 Project Structure

```
credit-risk-predictor/
├── api/
│   └── main.py                  ← FastAPI REST endpoint (local dev)
├── dashboard/
│   └── app.py                   ← Streamlit dashboard (deployed)
├── notebooks/
│   ├── 01_eda.ipynb             ← Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── model.pkl                ← Trained XGBoost model
│   └── feature_names.pkl        ← Feature engineering pipeline
├── assets/                      ← Screenshots for README
├── .gitignore
├── README.md
├── requirements.txt
└── .streamlit/
    └── config.toml              ← Streamlit configuration
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **ML Models** | XGBoost, LightGBM, Scikit-learn |
| **Imbalance Handling** | SMOTE (imbalanced-learn) |
| **Explainability** | SHAP (Shapley Additive exPlanations) |
| **Experiment Tracking** | MLflow |
| **API Framework** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Version Control** | Git, GitHub |
| **Deployment** | Streamlit Cloud |

---

## 💼 Key Achievements

✅ Built end-to-end ML pipeline on 307K+ loan records achieving **ROC-AUC 0.73** and **Recall 0.68** using XGBoost with SMOTE

✅ Engineered 4 domain-specific features and applied one-hot encoding across 10 categorical variables, expanding feature space from 122 → 193 dimensions

✅ Optimized decision threshold at 0.1 based on business cost matrix (false negative 10x costly), improving defaulter detection by 15% over baseline

✅ Implemented MLflow experiment tracking with full parameter/metric logging, demonstrating production ML workflow awareness

✅ Developed FastAPI REST endpoint with request validation and integrated Streamlit dashboard with SHAP explainability for credit analyst workflow

✅ Deployed live Streamlit application enabling real-time loan default risk predictions with feature importance visualizations

---

## 🔮 Future Enhancements

- [ ] Containerize FastAPI endpoint (Docker) for production deployment
- [ ] Add real-time data ingestion from loan application sources
- [ ] Implement automated retraining pipeline with drift detection
- [ ] Add A/B testing framework for threshold optimization
- [ ] Extend to multi-class risk scoring (low/medium/high risk)

---

## 👩‍💻 Author

**Priya Kumari**  
[GitHub](https://github.com/PriyaKumari2002) • [LinkedIn](https://linkedin.com/in/priya374)

Built as a portfolio project simulating real-world BFSI credit risk assessment workflows.

---

## 📄 License

This project is open source and available under the MIT License.

⭐ If you found this project useful, please give it a star!
