from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_data(X, y):
    """
    Takes x and y values and splits them into train and test data.
    After splitting the data, it standardizes the features.
    Returns the split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75, test_size=0.25)
    
    # Initialize scaler
    scaler = StandardScaler()
    # Fit train and test data into scaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def feature_selection(model, X, y):
    """
    Features will be selected using Recursive Feature Elimination.
    Returns the RFE variable which contains the selected features.
    """
    
    # Initialize RFE
    rfe = RFE(estimator=model, n_features_to_select=5)
    
    # Fit x and y into RFE
    rfe.fit(X, y)
    
    return rfe