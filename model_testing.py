from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Add OneHotEncoder here
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math

# Load the dataset
data = pd.read_csv('data/cleaned_dataset.csv')

# Define features and target
features = ['Building Size - GIA (M2)', 'Property Type', 'Effective Year']
target = 'Base Price'

# Split data into features and target variable
X = data[features]
y = data[target]

# Check for NaN or infinite values in target variable
print(f"Any NaN in y: {y.isna().any()}")
print(f"Any infinite in y: {np.isinf(y).any()}")

# Impute missing target values with median
y = y.fillna(y.median())

# Check again after imputation
print(f"Any NaN in y after imputation: {y.isna().any()}")
print(f"Any infinite in y after imputation: {np.isinf(y).any()}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline for both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values for numeric features
            ('scaler', StandardScaler())  # Scale numeric features
        ]), ['Building Size - GIA (M2)', 'Effective Year']),  # Columns to scale
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncode categorical features
        ]), ['Property Type'])  # Columns to encode
    ])

# Define the model pipeline with XGBoost
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor())  # You can change this to RandomForestRegressor() or any other model
])

# Set up GridSearchCV for hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200],  # Number of trees
    'regressor__learning_rate': [0.01, 0.1],  # Learning rate for XGBoost
    'regressor__max_depth': [3, 6, 10],  # Depth of trees
    'regressor__subsample': [0.8, 1.0]  # Subsample ratio
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_}")

# Make predictions
y_pred = grid_search.predict(X_test)

# Optionally, evaluate the model (e.g., using RMSE)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error after tuning: {rmse}')
