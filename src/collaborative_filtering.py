import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


def create_user_item_matrix(ratings: pd.DataFrame):
    user_item_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean" # In case there are two ratings for the same user and movie
    )

    return user_item_matrix


def normalize_user_rating(user_item_matrix: pd.DataFrame):
    user_means = user_item_matrix.mean(axis=1)

    normalized_matrix = user_item_matrix.sub(user_means, axis=0)

    normalized_matrix = normalized_matrix.fillna(0) # 0 means 'without information / neutral rating'

    return normalized_matrix, user_means


def train_svd_model(
    normalied_matrix: pd.DataFrame,
    n_components: int = 50,
    random_state: int = 42
):
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state
    )

    user_factors = svd.fit_transform(normalied_matrix) # User represented by their latent tastes
    item_factors = svd.components_ # Movies represented by their latent characteristics

    return svd, user_factors, item_factors


def predict_ratings(
    user_factors,
    item_factors,
    user_means: pd.Series,
    user_item_matrix: pd.DataFrame
) -> pd.DataFrame:
    predicted_normalized = np.dot(user_factors, item_factors)

    predicted_ratings = predicted_normalized + user_means.values.reshape(-1, 1)

    predicted_ratings_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )

    return predicted_ratings_df


def recommend_movies_for_user(
    user_id: int,
    predicted_ratings: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int
) -> pd.DataFrame:
    return
