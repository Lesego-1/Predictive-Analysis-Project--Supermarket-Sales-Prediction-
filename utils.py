import pandas as pd 
import sqlite3 as sq

def store_raw_data(input_file):
    """
    This fuction reads a csv input file and stores it as raw unprocessed data in sqlite format.
    """
    # Connect to database
    conn = sq.connect("data.sqlite")
    
    # Load raw data
    df = pd.read_csv(input_file, parse_dates=True)
    
    # Save dataFrame as sqlite table
    df.to_sql("raw_data", conn, index=False, if_exists="replace")
    
    # Close database connection
    conn.close()
    
def store_clean_data(input_file):
    """
    Stores the processed data into a sqlite databse.
    """
    # Connect to the database
    conn = sq.connect("data.sqlite")
    
    # Load processed data
    df = pd.read_csv(input_file)
    
    # Save data into the database
    df.to_sql("cleaned_data", conn, index=False, if_exists="replace")
    
    # Close database connection
    conn.close()
    
def load_data(database, table):
    """
    Loads a table from a database to a pandas dataframe.
    """
    # Connect to database
    conn = sq.connect(database)
    
    # Load data into dataframe
    df = pd.read_sql(f'SELECT * FROM {table}', conn)
    
    # Close the database connection
    conn.close()
    
    return df