# 🚨 FraudWatch: Real-Time Credit Card Fraud Detection using XGBoost, SMOTE & Flask

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge\&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Fraud_Detection-orange?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Backend_API-black?style=for-the-badge\&logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Production_Ready-success?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Highly_Imbalanced-purple?style=for-the-badge)

---

# 📌 Project Overview

FraudWatch is an AI-powered real-time credit card fraud detection system designed to identify suspicious financial transactions using machine learning.

The system combines:

* XGBoost classification
* SMOTE imbalance handling
* Feature scaling & preprocessing pipelines
* Flask backend APIs
* Interactive frontend integration

to detect fraudulent transactions with high accuracy and strong fraud recall on highly imbalanced financial datasets.

---
🔗WEBSITE

🔗 https://fraudwatch-wb0j.onrender.com

---
# 🎯 Key Highlights

✅ Achieved **99.93% Accuracy**

✅ Achieved **97.49% ROC-AUC**

✅ Achieved **82.76% Fraud Recall**

✅ Built complete end-to-end ML pipeline

✅ Real-time fraud prediction API using Flask

✅ Trained on real-world highly imbalanced financial transaction data

---

# 🧠 Machine Learning Pipeline

## 🔹 Data Processing

The model was trained on anonymized credit card transaction data containing:

* transaction time
* transaction amount
* PCA-transformed features (V1–V28)

The dataset is highly imbalanced, making fraud detection significantly challenging.

---

## 🔹 Feature Engineering & Preprocessing

Pipeline includes:

* Median imputation
* Standard scaling
* Stratified train-test splitting
* Amount-based stratification

Implemented using Scikit-learn Pipelines for production consistency.

---

## 🔹 Imbalance Handling using SMOTE

Fraud datasets suffer from extreme class imbalance.

Original fraud ratio:

```txt
~0.17% Fraud Transactions
```

To solve this:

* SMOTE oversampling was applied
* Minority fraud class was synthetically balanced

After SMOTE:

```txt
Class 0 → 227469
Class 1 → 227469
```

This significantly improved recall and fraud detection sensitivity.

---

# ⚡ Model Architecture

## XGBoost Classifier

The system uses XGBoost due to:

* superior performance on tabular data
* strong imbalance handling
* high interpretability
* fast inference speed

Key hyperparameters:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.9,
    gamma=0.1,
    min_child_weight=5
)
```

---

# 📊 Model Performance

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 99.93% |
| ROC-AUC  | 97.49% |
| Recall   | 82.76% |
| F1 Score | 83.48% |

---

# 📈 Visual Analysis

## Fraud Distribution

Demonstrates severe class imbalance in real-world financial datasets.

## ROC Curve

Shows strong classifier separation capability with high AUC.

## Confusion Matrix

Illustrates low false-negative fraud detection.

## Feature Importance

Highlights the most influential transaction patterns learned by XGBoost.

---

# 🌐 Real-Time Fraud Detection API

The Flask backend exposes a prediction API for real-time fraud analysis.

## API Endpoint

```http
POST /predict
```

## Example Request

```json
{
  "time": 12345,
  "amount": 250,
  "features": [
    0.1, -1.2, 0.5, 0.9, -0.4,
    0.2, 1.1, -0.8, 0.7, -0.1,
    0.4, 0.2, -0.3, 1.4, 0.9,
    -1.0, 0.5, 0.6, -0.7, 1.2,
    0.3, -0.9, 0.8, 0.1, -0.5,
    0.4, 0.6, -1.1
  ]
}
```

## Example Response

```json
{
  "prediction": 1,
  "label": "Fraud",
  "fraud_probability": 97.81
}
```

---

# 💻 Frontend Features

The project includes a responsive frontend interface with:

* live transaction prediction
* interactive fraud probability display
* modern UI/UX design
* animated fraud dashboard
* REST API integration

---

# 📂 Project Structure

```bash
FraudWatch/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── runtime.txt
│
├── models/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── plots/
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── fraud_distribution.png
│   └── confusion_matrix.png
│
└── data/
    └── creditcard.csv
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/fraudwatch-ai.git

cd fraudwatch-ai
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train Model

```bash
python train.py
```

---

# ▶️ Run Flask App

```bash
python app.py
```

---

# 📦 Requirements

Main libraries used:

* Flask
* XGBoost
* Scikit-learn
* Pandas
* NumPy
* Imbalanced-learn
* Joblib
* Matplotlib
* Seaborn

---

# 🔬 Research & Engineering Concepts Used

* Supervised Machine Learning
* Imbalanced Learning
* SMOTE Oversampling
* Fraud Detection Systems
* REST APIs
* Feature Scaling
* Model Serialization
* Production ML Pipelines
* Flask Backend Development

---

# ⚠️ Limitations

* PCA-transformed features reduce interpretability
* Fraud patterns evolve over time
* Real banking systems require continuous retraining
* Threshold tuning depends on business requirements
* False positives remain a challenge in high-security systems

---

# 🚀 Future Improvements

* Real-time streaming transaction analysis
* Deep learning anomaly detection
* SHAP explainability integration
* Docker container deployment
* Kafka-based fraud streaming pipeline
* Real bank transaction integrations

---

# 📚 Dataset

Dataset sourced from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

# 👨‍💻 Author

Manvith Mogaveera

Electronics & Telecommunication Engineering student exploring:

* Artificial Intelligence
* Machine Learning
* NLP
* Fraud Detection Systems
* Healthcare AI
* Intelligent Analytics Platforms

---

# ⭐ Final Note

This project demonstrates:

* end-to-end ML engineering
* imbalance-aware learning
* real-time API integration
* production-ready preprocessing pipelines
* practical fraud detection workflows

designed to simulate real-world financial fraud detection systems.
