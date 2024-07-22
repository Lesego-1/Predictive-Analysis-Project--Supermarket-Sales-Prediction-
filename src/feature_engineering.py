from sklearn.feature_selection import RFE

def feature_selection(model, x, y):
    """
    Features will be selected using Recursive Feature Elimination.
    Returns the RFE variable which contains the selected features.
    """
    
    # Initialize RFE
    rfe = RFE(estimator=model, n_features_to_select=5)
    
    # Fit x and y into RFE
    rfe.fit(x, y)
    
    return rfe