import pandas as pd

def load_data(file_path):
    """
    This function loads data from a filepath and returns a pandas DataFrame containing the data.
    """
    # Convert file into a Pandas DataFrame
    df = pd.read_csv(file_path, parse_dates=True)
    
    return df

def clean_data(df):
    """
    Takes a Pandas DataFrame and cleans the data to be ready for model training.
    Returns the cleaned DataFrame.
    """
    
    # Calculate total missing data
    total_missing_data = df.isnull().sum().sum()
    
    if total_missing_data == 0:
        # There is no missing data
        pass
    else:
        # Fill out missing data
        print(f"Total missing values: {total_missing_data}")
        df.fillna(df.mean())
        
    # One hot encode data
    df = pd.get_dummies(df, drop_first=False)
    
    return df

def save_data(df, file_path):
    """
    This function saves the processed DataFrame into a file.
    """
    
    # Save the csv to the selected file
    df.to_csv(file_path, index=False)