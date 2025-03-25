## Sprint 3: Model Development and Evaluation

### Overview

This repository contains the Sprint 3: Model Development and Evaluation for the Property Price Prediction project. The aim of this sprint is to develop, tune, and compare different machine learning models to predict property prices. The repository contains Jupyter notebooks for model development, evaluation, and comparison along with a Python script for testing the final model.

### File Structure

```bash
/sprint3
 ├── data/                       # Dataset folder (cleaned property dataset)
 │    └── cleaned_dataset.csv    # Cleaned property dataset
 ├── Index.ipynb                 # Main Jupyter Notebook (Model Development & Evaluation)
 ├── main.ipynb                  # Alternative ML Model (Ridge Regression)
 ├── model_testing.py            # Script for testing the final model
 ├── property_price_model.pkl    # Best XGBoost model saved after hyperparameter tuning
 ├── property_price_model_2.pkl  # Alternative model (not required for testing)
 └── README.md                   # Project documentation
```

### Description

#### 1. Index.ipynb (Primary Notebook)

- The core notebook for model development and evaluation.
- Includes the full workflow:
  - Data preprocessing
  - Model training (XGBoost, Random Forest, Lasso, Linear Regression)
  - Hyperparameter tuning
  - Model evaluation with comparison plots between models.
- The final model is saved as property_price_model.pkl and used for testing.

#### 2. Main.ipynb (for alternative model)

- Explores **`Ridge Regression`** as an alternative model.
- Includes:
  - Model training with **`Ridge Regression`**.
  - Comparison plots between Ridge and other models.
- The model is saved as **`property_price_model_2.pkl`** but is not required for testing.

#### 3. model_testing.py

- Script for testing the final model (**`property_price_model.pkl`**).
- Performs predictions on test data using the trained model.
- Only the **`XGBoost model`** from `Index.ipynb` is tested.
- The alternative **`Ridge Model`** is excluded from the testing phase.

#### 4. property_price_model.pkl

- Best-performing XGBoost model saved after hyperparameter tuning.
- This is the model used in **`model_testing.py`** for predictions.

#### 5. property_price_model_2.pkl

- The alternative model
- not used for testing but saved for further development

### Usage Instructions

#### 1. Run the Model Development

- Open and run **`Index.ipynb`** for the full model development and evaluation process.
- Use **`Main.ipynb`** to explore `Ridge Regression` as an alternative model.

#### 2. Test the Final Model:

- Execute **`model_testing.py`** to test the XGBoost model with new property data.
- Ensure the **`property_price_model.pkl`** is present in the repository

### Future Improvements

- Add more feature (e.g., property age, neighborhood score) to enhance accuracy.
- Implement ensemble learning for better model performance.
- Include R2 score in the evaluation metrics for comprehensive analysis.

### Author

- Pyae Linn (24045065)
- MSc Data Science Student at UWE
