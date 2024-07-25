from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def evaluate_model(y_true, y_pred):
    """
    Evaluates a given model by printing out the mean squared error and R2 score.
    """
    # Calculate mean squared error
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate R2 score
    r_score = r2_score(y_true, y_pred)
    
    # Print out results
    print(f"Mean Squared Error: {round(mse, 2)}")
    print(f"R2 Score: {round(r_score, 2)}")