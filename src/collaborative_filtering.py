import pandas as pd


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
    return
