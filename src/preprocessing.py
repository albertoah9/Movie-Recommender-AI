import pandas as pd

def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the movies dataset"""

    movies = movies.copy()

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")

    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    movies["genres_list"] = movies["genres"].str.split("|")

    return movies

def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the ratings dataset"""

    ratings = ratings.copy()

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    return ratings