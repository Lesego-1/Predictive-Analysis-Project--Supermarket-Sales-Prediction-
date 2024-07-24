from src.data_preprocessing import load_data, clean_data, save_data
from src.feature_engineering import split_data
from src.model import linear_regression_model, ridge_regression_model
from src.model import random_forest_regression_model
from src.evaluation import evaluate_model
from src.utils import store_clean_data, store_raw_data
import pandas as pd

# Initialize paths to data files
raw_data_file = r"data\raw_data.csv"
clean_data_file = r"data\processed_data.csv"

# Load data into dataFrame
df = load_data(raw_data_file)

# Process the data
df = clean_data(df)

# Save data to a processed data file
save_data(df, clean_data_file)

# Get target names
target_names = ['gross margin percentage', 'gross income', 'Rating']

# Initialize features and targets
X = df.drop(columns=['gross margin percentage', 'gross income', 'Rating'])
y = df[target_names]

# Slit data into train and test
X_train, X_test, y_train, y_test = split_data(X, y)

# Get predicted values for all models
linear_regression_values = linear_regression_model(X_train, X_test, y_train)
ridge_regression_values = ridge_regression_model(X_train, X_test, y_train)
random_forest_values = random_forest_regression_model(X_train, X_test, y_train)

# Evaluate models
print("Linear Regression Evaluation")
evaluate_model(y_test, linear_regression_values)

print("\nRidge Regression Evaluation")
evaluate_model(y_test, ridge_regression_values)

print("\nRandom Forests Regression Evaluation")
evaluate_model(y_test, random_forest_values)

# Store raw data into sqlite database
store_raw_data(raw_data_file)

# Store clean data into database
store_clean_data(clean_data_file)