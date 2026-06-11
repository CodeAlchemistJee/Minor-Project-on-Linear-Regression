# Machine Learning from Scratch: Real Estate Price Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Math-013243.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)

This repository contains a comprehensive, multi-stage implementation of foundational Machine Learning regression algorithms built **entirely from scratch** (without the use of `scikit-learn`). 

The project focuses on predicting real estate prices based on various property features, transitioning from basic univariate analysis to advanced polynomial expansions, regularization techniques, and final meta-model stacking.

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Key Features & Algorithms](#key-features--algorithms)
4. [Installation & Requirements](#installation--requirements)
5. [Usage](#usage)

## 🎯 Project Overview
The objective of this project is to demonstrate a deep mathematical understanding of how machine learning models learn. By writing gradient descent algorithms, matrix operations, cross-validation loops, and statistical tests natively in Python, this project explores the underlying mechanics of statistical modeling, feature engineering, the bias-variance tradeoff, and model ensembling.

## 📂 Repository Structure

The project is divided into four distinct phases of increasing complexity:

### `Part_1_Simple_Linear_Regression.py`
Focuses on establishing the baseline relationship between property Area and Price.
* **Algorithm:** Custom Gradient Descent (Simple Linear Regression).
* **Data Processing:** Mean/Median imputation and Interquartile Range (IQR) outlier removal.
* **Diagnostics:** 95% Confidence Intervals calculated via standard error, and residual homoscedasticity plotting.

### `Part_2_Multiple_Linear_Regression.py`
Expands the model to include multiple dimensions (Bedrooms, Bathrooms, Age, Neighborhoods).
* **Algorithms:** Normal Equation (Matrix Inverse) and Vectorized Gradient Descent.
* **Regularization:** Native implementation of L1 (Lasso) and L2 (Ridge) penalties.
* **Feature Engineering:** One-Hot Encoding for categoricals, feature interactions (`area_x_bedrooms`).
* **Validation:** Custom 5-Fold Cross-Validation and Variance Inflation Factor (VIF) multicollinearity checks.

### `Part_3_Polynomial_Regression.py`
Introduces non-linear modeling to capture complex market dynamics (e.g., diminishing returns on acreage).
* **Algorithm:** Polynomial Feature Expansion (Up to Degree 5).
* **Optimization:** Early Stopping logic based on validation set performance.
* **Model Selection:** Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) scoring.
* **Advanced Features:** Piecewise Polynomials (Splines) and Kernel Ridge Regression (Polynomial Kernel) using matrix operations.

### `Part_4_Model_Selection_And_Stacking.py`
Focuses on rigorous statistical evaluation and combining previous models into a high-performance ensemble.
* **Evaluation Metrics:** Custom implementations of R², Adjusted R², RMSE, MAE, and MAPE.
* **Statistical Rigor:** Bootstrap Confidence Intervals for RMSE and Paired T-Tests for significance testing.
* **Ensembling:** Stacking Meta-Learner that combines Ridge Regression and Polynomial Regression predictions.
* **Business Application:** Generation of 95% Prediction Intervals for automated valuation model (AVM) confidence bounds.

## ✨ Key Features & Algorithms Built from Scratch
* **Gradient Descent Engine:** Complete with learning rate scheduling/decay and tolerance-based convergence criteria.
* **Z-Score Standardization:** Crucial for preventing gradient explosion in high-degree polynomial matrices.
* **K-Fold CV Generator:** Custom indexing and shuffling logic for robust model evaluation.
* **AIC / BIC Calculators:** Mathematical implementations penalizing model complexity to prevent overfitting.
* **Meta-Ensemble Stacking:** Combining multiple base models and training a meta-learner on their predictions to optimize accuracy.
* **Statistical Testing Suite:** Bootstrapping for confidence intervals and paired T-tests for determining model significance.
* **Data Synthesizer:** Built-in numpy data generators to simulate realistic real estate markets with injected noise, missing values, and right-skewed outliers.

## ⚙️ Installation & Requirements

Because this project avoids high-level ML libraries, the dependency list is extremely lightweight. 

```bash
# Clone the repository
git clone [https://github.com/yourusername/ML-Regression-From-Scratch.git](https://github.com/yourusername/ML-Regression-From-Scratch.git)

# Navigate to the directory
cd ML-Regression-From-Scratch

# Install the required scientific computing libraries
pip install numpy pandas matplotlib scipy
