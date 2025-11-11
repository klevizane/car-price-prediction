# car-price-prediction

# 📈 Advanced Regression Modeling: Methodological Rigor and Error Optimization

This repository documents a comprehensive **predictive regression project** focused on intensive **Feature Engineering** and **error analysis**. The core objective was to demonstrate the ability to reduce the **Mean Absolute Error (MAE)** and enhance the overall robustness of model predictions when dealing with a noisy, real-world dataset.

---

## 1. Methodology and Data Engineering Process

A two-phase data processing pipeline was implemented to ensure the highest quality and robustness of the features:

### Phase 1: Initial Cleaning, Transformation, and Baseline Model

* Data Cleaning and Transformation: Handled inconsistencies in data types (e.g., converting mixed `str` and `int` columns) and standardized features.
* Missing Value Imputation Strategy:
    * Sophisticated Imputation: Null values in the cylinders column (165 records) were imputed using a Random Forest Regressor, trained on other features. This is a robust, advanced technique over simple mean/mode imputation.
    * Simple Imputation: Null values in horsepower were filled using the column's mean.
*  Model: An initial model was trained to establish performance metrics before advanced optimization.
    * Baseline MAE: **$6,018.44**
    * Baseline R²: **0.9461**
    * Baseline RMSE: **12,326.29**

### Phase 2: Advanced Optimization, Feature Engineering, and Feature Aggregation

Analysis of the baseline model residuals revealed that the model struggled with high-value points, suggesting missing discriminatory features and high variance.

* Logarithmic Transformation: A **logarithmic scale** was applied to the target variable price to mitigate the impact of extreme *outliers* and normalize the distribution, improving model linearity.
* Handling Duplicates and Inconsistencies:
    * Duplicate rows were eliminated.
    * Aggregation Strategy: Rows with identical car characteristics but differing prices (indicating missing latent features) were grouped. The price was then replaced with the mean price of the group, introducing a more robust, aggregated price estimate.
* Strategic Feature Engineering: A new categorical feature car_class was created to classify the **vehicle segment/range**, providing the model with essential information to differentiate between price bands.

### 2. Key Quantitative Results and Final Optimization

The systematic feature engineering effort led to a significant performance boost, which was then fine-tuned via hyperparameter optimization.

| Metric | Initial Baseline | Post-Feature Engineering | Optimized Model (GridSearchCV) | Total MAE Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **R²** | 0.9461 | 0.9541 | **0.9622** | +1.61 pts |
| **RMSE** | $12,326.29 | $9,707.57 | **$8,810.43** | **-28.52%** |
| **MAE** | $6,018.44 | $4,448.44 | **$4,179.45** | **-30.55%** |

* Hyperparameter Tuning: GridSearchCV was utilized to find the optimal set of hyperparameters for the final model XGBoost, ensuring the model achieved its best possible performance and maintained statistical robustness.

