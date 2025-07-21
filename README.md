# House Prices - Advanced Regression Techniques (Kaggle Top 13%)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Top%2013%25-20BEFF.svg)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)

This repository contains a comprehensive solution for the Kaggle competition "House Prices: Advanced Regression Techniques". The goal of this project is to predict the final sale price of homes in Ames, Iowa, based on 79 explanatory variables.

This project goes beyond a simple baseline, employing a robust pipeline of advanced feature engineering, systematic model evaluation, and hyperparameter tuning to achieve a competitive score.

### Final Result
* **Kaggle Public LB Score:** **0.12450**
* **Ranking:** **Top 13%** (~650th out of 5,000+ teams)

### Tech Stack
* Python 3
* Pandas & NumPy for data manipulation
* Scikit-learn for preprocessing and modeling pipelines
* LightGBM, XGBoost, and CatBoost for gradient boosting models
* Matplotlib & Seaborn for data visualization

### Methodology / Workflow

The solution was developed through a structured, iterative process:

1.  **Data Cleaning & Outlier Removal:** The process began with initial data cleaning, including the removal of two well-known outliers in the `GrLivArea` feature, which significantly improved model stability. The target variable, `SalePrice`, was also transformed using `log1p` to handle its right-skewed distribution.

2.  **Advanced Feature Engineering:** A comprehensive feature engineering function was created to enrich the dataset. Key techniques included:
    * **Smart Imputation:** Interpreting `NaN` values based on context (e.g., `NaN` in `PoolQC` means 'No Pool', not missing data).
    * **Ordinal Feature Mapping:** Converting categorical features with inherent order (e.g., `ExterQual`: 'Excellent', 'Good') into numerical rankings.
    * **Feature Creation:** Generating new, high-value features like `TotalSF` (total square footage), `HouseAge`, and interaction features (e.g., `OverallQual * TotalSF`).
    * **Skewness Transformation:** Applying a `log1p` transform to all numerical features with high skewness to normalize their distributions.

3.  **Modeling & Hyperparameter Tuning:** Three top-tier gradient boosting models were systematically evaluated: **LightGBM**, **XGBoost**, and **CatBoost**. `GridSearchCV` was used to find the optimal set of hyperparameters for each model.

4.  **Robust Validation Strategy:** Instead of relying on a single train-validation split (which can be misleading), a **5-Fold Cross-Validation** framework was implemented. This provided a much more reliable estimate of each model's true performance on unseen data. The tuned `CatBoost` model emerged as the single best performer with a CV mean score of **~0.11762**.

5.  **Final Prediction:** The champion model (tuned CatBoost) was retrained on the **entire** cleaned training dataset to maximize its learning before predicting on the final test set.

### How to Run
1.  Clone this repository: `git clone https://github.com/Luo0105/house_price.git`
2.  Ensure you have the required libraries installed.
3.  Place the competition data (`train.csv`, `test.csv`) in a designated folder.
4.  Run the Jupyter Notebook to see the full analysis and reproduce the results.
