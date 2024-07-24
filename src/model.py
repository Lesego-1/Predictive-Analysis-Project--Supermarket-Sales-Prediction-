from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def linear_regression_model(X_train, X_test, y_train, y_test):
    """
    Builds a linear regression model to predict gross income.
    Returns the model, and predicted y values.
    """
    # Initialize model
    model = LinearRegression()
    # Fit X and y data into model
    model.fit(X_train, y_train)
    
    # Predict y values
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return model, y_train_pred, y_test_pred

def ridge_regression_model(X_train, X_test, y_train):
    """
    Creates a ridge regression model.
    GridSearchCV is used to find the best alpha value for the model.
    Returns the model with the best alpha value and the predicted values
    """
    # Initialize model
    model = Ridge(alpha=1.0)
    
    # Create param grid
    param_grid = {
        'alpha' : [0.1, 0.5, 1.0, 1.5, 10, 100]
    }
    
    # Initialize model with best alpha value
    best_model = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
    
    # Fit data into model
    best_model.fit(X_train, y_train)
    
    # Predict y values
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    return best_model, y_train_pred, y_test_pred

def lasso_regression_model(X_train, X_test, y_train):
    """
    Creates a lasso regression model.
    GridSearchCV is used to find the best alpha value for the model.
    Returns the lasso regression model and the predicted values.
    """
    # Initialize model
    model = Lasso(alpha=1.0)
    
    # Create param grid
    param_grid = {
        'alpha' : [0.1, 0.5, 1.0, 1.5, 10, 100]
    }
    
    # Initialize model with best alpha value
    best_model = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
    
    # Fit data into model
    best_model.fit(X_train, y_train)
    
    # Predict y values
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    return best_model, y_train_pred, y_test_pred

def random_forest_regress_model(X_train, X_test, y_train):
    """
    Creates a random forests regression model and returns the model along with the predicted values.
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
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    return best_model, y_train_pred, y_test_pred

def support_vector_regressor_model(X_train, X_test, y_train):
    """
    Creates a SVR model and returns the model and predicted y values.
    Uses GridSearchCV to find the best kernels to use for the model.
    """
    # Initialize model
    model = SVR(kernel='linear')
    
    # Create parameter grid
    param_grid = {
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    }
    
    # Create model with the best kernel
    best_model = GridSearchCV(model, cv=5, n_jobs=-1, param_grid=param_grid)
    
    # Fit data into the best model
    best_model.fit(X_train, y_train)
    
    # Predict y values
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    return best_model, y_train_pred, y_test_pred