import pandas as pd

def load_movies(path: str) -> pd.DataFrame:
    """Load the movies dataset from a CSV file"""
    return pd.read_csv(path)

def load_ratings(path:str) -> pd.DataFrame:
    """Load the ratings dataset from a CSV file"""
    return pd.read_csv(path)
