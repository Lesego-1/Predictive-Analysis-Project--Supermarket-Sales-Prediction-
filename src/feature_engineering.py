from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_data(X, y):
    """
    Takes x and y values and splits them into train and test data.
    Returns the split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75, test_size=0.25)
    
    return X_train, X_test, y_train, y_test