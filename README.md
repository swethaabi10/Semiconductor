# Semiconductor Manufacturing Yield Prediction

Predicting semiconductor manufacturing process quality outcomes (**Pass** vs. **Fail**) using real-time machine sensor telemetry data and supervised machine learning classification models.

---

## 📌 Project Overview

In semiconductor manufacturing, early detection of line faults and defective chips is critical for operational efficiency. This project implements an end-to-end Machine Learning pipeline to classify production batches based on high-dimensional sensor metrics.

### Key Objectives
- **Automate Quality Control**: Classify production quality into Pass (`-1`) or Fail (`1`).
- **Preprocess Telemetry Data**: Clean and scale 590+ numerical sensor signals.
- **Address Class Imbalance**: Utilize SMOTE oversampling to effectively learn minority failure patterns.
- **Model Deployment**: Export the top-performing model for production batch inference.

---

## 📊 Dataset Overview

- **Source File**: `signal-data.csv`
- **Observations**: 1,567 rows
- **Attributes**: 592 columns
  - **Timestamp**: `Time`
  - **Sensor Features**: `0` through `589`
  - **Target Label**: `Pass/Fail` (`-1` = Pass, `1` = Fail)

### Class Distribution
| Target Label | Quality Outcome | Observations | Percentage |
| :--- | :--- | :--- | :--- |
| **-1** | Pass | 1,463 | ~93.4% |
| **1** | Fail | 104 | ~6.6% |

---

## ⚙️ Data Preprocessing & Workflow

1. **Missing Value Imputation**: Handled missing telemetry entries by zero-filling (`.fillna(0)`).
2. **Zero-Variance Feature Removal**: Stripped constant columns, reducing feature dimensions from 592 down to 480 active sensor columns.
3. **Exploratory Dimensionality Reduction (PCA)**:
   - **PC1**: Captures ~57.39% of variance.
   - **PC2**: Captures ~23.37% of variance.
   - **Combined 2D Variance**: ~80.76%.
4. **Feature Normalization**: Scaled independent features to mean zero and unit variance using `StandardScaler`.
5. **Class Resampling**: Synthetic Minority Over-sampling Technique (SMOTE) applied to balance failure representation during training.

---

## 🚀 Model Performance Comparison

Models were trained on an 80/20 train-test split (`random_state=42`) and evaluated on holdout test data.

| Algorithm | Test Accuracy | Status |
| :--- | :--- | :--- |
| **Support Vector Machine (SVM)** | **99.15%** | 🏆 Top Model |
| **Random Forest Classifier** | **98.98%** | 🥈 Runner Up |
| **Gaussian Naive Bayes** | **56.31%** | ❌ Baseline |

> **Key Insight**: SVM achieved optimal decision boundaries by leveraging scaled feature spaces and synthetic class balancing.

---

## 📁 Repository Structure

```text
.
├── Manufacturing Yield Prediction Project Documentation.pdf  # Comprehensive project report
├── README.md                                                 # Repository documentation
├── Yieldproj.ipynb                                           # Jupyter Notebook with full code
└── signal-data.csv                                           # Telemetry sensor dataset
