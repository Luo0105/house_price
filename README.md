# House Prices - Advanced Regression Techniques (Kagle Top 13%)

[![Kaggle Rank](https://img.shields.io/badge/Kaggle-Top%2013%25-20BEFF.svg)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-%23584ACB.svg?logo=catboost&logoColor=white)](https://catboost.ai/)

This repository contains a comprehensive solution for the Kaggle competition "House Prices: Advanced Regression Techniques". The goal of this project is to predict the final sale price of homes in Ames, Iowa.

This project goes beyond a simple baseline, employing a robust pipeline of advanced feature engineering, systematic model evaluation with cross-validation, and hyperparameter tuning to achieve a competitive score.

### Final Result
* **Kaggle Public LB Score:** **0.12450**
* **Ranking:** **Top 13%** (~650th out of 5,000+ teams)

### Project Workflow

The entire project follows a structured, iterative process as illustrated below:

```mermaid
graph TD
    %% A. Data Preparation
    A["Load Data (train.csv & test.csv)"] --> B["Clean Data (Remove Outliers)"];
    B --> C["Prepare Full Dataset (X_full & y_log)"];

    %% B. Feature Engineering Subgraph
    C --> FE;
    subgraph FE [Advanced Feature Engineering]
        FE1["Smart Imputation<br>(e.g., NaN => 'None')"];
        FE2["Ordinal Mapping<br>(e.g., 'Ex' => 5)"];
        FE3["Type Conversion<br>(e.g., MSSubClass => Category)"];
        FE4["Feature Creation<br>(e.g., TotalSF, HouseAge)"];
        FE5["Interaction Features<br>(e.g., Qual x TotalSF)"];
        FE6["Skewness Transformation<br>(Log1p)"];
    end

    %% C. Modeling & Evaluation Subgraph
    FE --> ME;
    subgraph ME [Modeling & Evaluation]
        ME1("Start 5-Fold Cross-Validation");
        ME2{"Evaluate Models<br>RF, Ridge, LGBM, XGB, CatBoost"};
        ME3["Compare CV Mean & Std Dev"];
        ME4("Select Best Model: CatBoost<br>(CV Mean: ~0.117)");
        ME1 --> ME2 --> ME3 --> ME4;
    end
    
    %% D. Final Prediction
    ME4 --> F["Retrain Best Model<br>on 100% of Training Data"];
    F --> G["Apply Same Feature Engineering<br>to Test Set"];
    G --> H["Generate Predictions"];
    H --> I["Create submission.csv"];

    %% Optional Styling
    style FE fill:#f9f,stroke:#333,stroke-width:2px
    style ME fill:#ccf,stroke:#333,stroke-width:2px
```

### Key Methodologies

1.  **Data Cleaning & Outlier Removal:** The process began with initial data cleaning, including the removal of two well-known outliers in the `GrLivArea` feature. The target variable, `SalePrice`, was also transformed using `log1p` to handle its right-skewed distribution.

2.  **Advanced Feature Engineering:** A comprehensive feature engineering function was created to enrich the dataset. Key techniques included:
    * **Smart Imputation:** Interpreting `NaN` values based on context (e.g., `NaN` in `PoolQC` means 'No Pool').
    * **Ordinal Feature Mapping:** Converting categorical features with inherent order (e.g., `ExterQual`: 'Excellent', 'Good') into numerical rankings.
    * **Feature Creation & Interaction:** Generating new, high-value features like `TotalSF`, `HouseAge`, and interaction features (e.g., `OverallQual * TotalSF`).
    * **Skewness Transformation:** Applying a `log1p` transform to all numerical features with high skewness.

3.  **Modeling & Hyperparameter Tuning:** Five models, including three top-tier gradient boosting models (**LightGBM**, **XGBoost**, and **CatBoost**), were systematically evaluated. `GridSearchCV` was used to find the optimal set of hyperparameters for the boosting models.

4.  **Robust Validation Strategy:** A **5-Fold Cross-Validation** framework was implemented to provide a reliable estimate of each model's true performance on unseen data.

### Model Performance Comparison

The following chart summarizes the performance and stability of all evaluated models based on the 5-fold cross-validation. The tuned **Ridge** and **CatBoost** models emerged as the top performers.

![Model Performance Comparison Chart](images/model_comparison_with_std_labels.png)

### Final Prediction Strategy

The champion model from the cross-validation analysis (tuned CatBoost) was retrained on the **entire** cleaned training dataset to maximize its learning before predicting on the final test set.

### How to Run
1.  Clone this repository: `git clone https://github.com/Luo0105/house_price.git`
2.  Ensure you have the required libraries installed.
3.  Place the competition data (`train.csv`, `test.csv`) in a designated folder.
4.  Run the Jupyter Notebook (`house-price-v2.ipynb`) to see the full analysis and reproduce the results.
