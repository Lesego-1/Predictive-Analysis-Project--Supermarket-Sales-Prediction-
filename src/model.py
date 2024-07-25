from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

def linear_regression_model(X_train, X_test, y_train):
    """
    Builds a linear regression model to predict gross income.
    Returns the predicted y values.
    """
    # Initialize model
    model = LinearRegression()
    # Fit X and y data into model
    model.fit(X_train, y_train)
    
    # Predict y values
    y_test_pred = model.predict(X_test)
    
    return y_test_pred

def ridge_regression_model(X_train, X_test, y_train):
    """
    Creates a ridge regression model.
    GridSearchCV is used to find the best alpha value for the model.
    Returns the predicted values
    """
    # Initialize model
    model = Ridge(alpha=1.0, max_iter=10000)
    
    # Create param grid
    param_grid = {
        'alpha' : [0.1, 0.5, 1.0, 1.5, 10, 100]
    }
    
    # Initialize model with best alpha value
    best_model = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
    
    # Fit data into model
    best_model.fit(X_train, y_train)
    
    # Predict y values
    y_test_pred = best_model.predict(X_test)
    
    return y_test_pred

def random_forest_regression_model(X_train, X_test, y_train):
    """
    Creates a random forests regression model and returns the predicted values.
    GridSearchCV is used to find the best number of decision trees within the forest.
    """
    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create parameter grid
    param_grid = {
        'n_estimators' : [10, 50, 100, 150]
    }
    
    # Initialize gridsearch
    best_model = GridSearchCV(model, n_jobs=-1, param_grid=param_grid, cv=5)
    
    # Fit data into the best model
    best_model.fit(X_train, y_train)
    
    # Predict train and test values
    y_test_pred = best_model.predict(X_test)
    
    return y_test_pred