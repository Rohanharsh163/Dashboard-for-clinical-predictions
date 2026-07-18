# 🏥 Clinical ML Pipeline Dashboard

An end-to-end Machine Learning pipeline for clinical prediction under temporal data shift. This project was developed as part of the **BITS F464 – Machine Learning** course and demonstrates the complete ML workflow, from data preprocessing and feature engineering to model training, evaluation, continual learning, and interactive visualization using Streamlit.

---

## 📌 Overview

Healthcare datasets evolve over time, making it important to evaluate how machine learning models perform on future data. This project builds an automated pipeline that:

- Processes Electronic Health Record (EHR) data
- Performs feature engineering
- Trains multiple machine learning models
- Evaluates model performance under temporal distribution shift
- Applies continual learning using newer data
- Provides an interactive dashboard for visualization and analysis

---

## ✨ Features

- Interactive Streamlit dashboard
- Automated data preprocessing pipeline
- Patient-level feature engineering
- Temporal train-test dataset splitting
- Exploratory Data Analysis (EDA)
- Multiple ML models
  - Decision Tree
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (Neural Network)
- Model comparison
- Continual learning evaluation
- ROC Curves
- Confusion Matrices
- Feature Importance Analysis
- Performance metrics visualization

---

## 📂 Dataset

The project uses the **Synthea-MIMIC Electronic Health Record (EHR)** dataset.

The pipeline integrates multiple healthcare tables including:

- Patients
- Conditions
- Encounters
- Observations
- Medications
- Procedures
- Allergies
- Devices
- Immunizations

These are merged into a patient-level dataset before model training.

---

## ⚙️ Machine Learning Pipeline

```
Raw EHR Tables
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Temporal Dataset Split
       │
       ├──────── Dataset 1 (Historical)
       │
       └──────── Dataset 2 (Current)
                │
                ▼
Model Training
       │
       ▼
Performance Evaluation
       │
       ▼
Continual Learning
       │
       ▼
Interactive Dashboard
```

---

## 📊 Feature Engineering

The pipeline generates patient-level features including:

- Age
- Gender Encoding
- Race Encoding
- Ethnicity Encoding
- Observation statistics
  - Mean
  - Standard Deviation
- Observation Counts
- Medication Counts
- Procedure Counts
- Allergy Counts
- Device Counts

---

## 🤖 Models Used

- Decision Tree Classifier
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

Each model is evaluated on:

- Historical test data
- Future (temporal) test data
- Continual learning performance after retraining

---

## 📈 Evaluation Metrics

The dashboard reports:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score
- Confusion Matrix
- Classification Report

---

## 🖥️ Dashboard

The Streamlit dashboard contains:

- Dataset Overview
- Exploratory Data Analysis
- Feature Engineering Summary
- Model Training Results
- Performance Comparison
- Continual Learning Analysis
- Feature Importance Visualization

---

## 🛠️ Tech Stack

**Languages**

- Python

**Libraries**

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Clinical-ML-Pipeline.git
cd Clinical-ML-Pipeline
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run dashboard.py
```

---

## 📁 Project Structure

```
Clinical-ML-Pipeline/
│
├── dashboard.py
├── requirements.txt
├── README.md
├── synthea-mimic/
│   └── csv/
├── images/
└── reports/
```

---

## 📚 Learning Outcomes

Through this project, we explored:

- Healthcare data preprocessing
- Feature engineering
- Temporal machine learning
- Model comparison
- Continual learning
- Interactive ML dashboards
- Clinical prediction workflows

---

## 👥 Team

- Sailesh Nichenametla
- Rohan Harshith Amarthaluri
- Dhanush Thirunagari
- Aniket Shukla

---

## 📜 License

This project was developed for academic purposes as part of the BITS Pilani Machine Learning course.
