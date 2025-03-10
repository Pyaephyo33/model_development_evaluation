from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math

# Load the dataset (replace with your actual dataset path)
data = pd.read_csv('data/cleaned_dataset.csv')

# Define features and target
features = ['Building Size - GIA (M2)', 'Property Type', 'Effective Year']
target = 'Base Price' 

# Split into features and target variable
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

# Define the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor())  # You can change this to RandomForestRegressor() or any other model
])

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Optionally, evaluate the model (e.g., using RMSE)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')
