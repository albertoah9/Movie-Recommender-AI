import re
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

def extract_decade(title: str) -> str:
    """
    Extract the release year from a movie title and convert it into a decade label.
    """
    match = re.search(r"\((\d{4})\)", str(title))

    if not match:
        return "decade_unknown"
    
    year = int(match.group(1))
    decade = (year // 10) * 10

    return f"{decade}s_decade"

def prepare_metadata_soup(
    movies: pd.DataFrame,
    tags: pd.DataFrame
) -> pd.DataFrame:
    """
    Create metadata soup columns for the content-based recommender.

    Creates:
    - decade
    - soup_genres
    - soup_tags
    """

    movies = movies.copy()
    tags = tags.copy()

    movies["decade"] = movies.apply(extract_decade)

    movies["genres_clean"] = (
        movies["genres"]
        .fillna("")
        .str.replace("|", " ", regex=False)
        .str.replace("(no genres listed)", "", regex=False)
        .str.lower()
    )

    movies["soup_genres"] = (
        movies["genres_clean"] + " " + movies["decade"]
    ).str.strip()

    tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()

    tags_grouped = (
        tags.groupby("movieId")["tag"]
        .apply(lambda values: " ".join(values))
        .reset_index()
        .rename(columns={"tag": "tags_joined"})
    )

    movies = movies.merge(tags_grouped, on="movieId", how="left")

    movies["soup_tags"] = movies["tags_joined"].fillna("")
    movies["soup_tags"] = movies["soup_tags"].str.strip()

    movies.loc[movies["soup_tags"] == "", "soup_tags"] = movies.loc[
        movies["soup_tags"] == "",
        "soup_genres"
    ]

    return movies

def calculate_weighted_rating(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    popularity_percentile: float = 0.70
) -> pd.DataFrame:
    """
    Calculate IMDb-style weighted rating and normalize it to [0, 1].

    WR = (v / (v + m)) * R + (m / (v + m)) * C

    Where:
    - v = number of ratings for the movie
    - R = average rating for the movie
    - C = global average rating
    - m = minimum number of votes required, based on percentile
    """

    movies = movies.copy()
    ratings = ratings.copy()

    rating_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(v="count", R="mean").reset_index()
    )

    C = ratings["rating"].mean()
    m = rating_stats["v"].quantile(popularity_percentile)

    rating_stats["weighed_rating"] = (
        (rating_stats["v"] / (rating_stats["v"] + m)) * rating_stats["R"]
        + (m / (rating_stats["v"] + m)) * C
    )

    scaler = MinMaxScaler(feature_range=(0, 1))

    rating_stats["quality_score"] = scaler.fit_transform(
        rating_stats[["weighted_rating"]]
    )

    movies = movies.merge(
        rating_stats[["movieId", "v", "R", "weighted_rating", "quality_score"]],
        on="movieId",
        how="left"
    )

    movies["v"] = movies["v"].fillna(0)
    movies["R"] = movies["R"].fillna(C)
    movies["weighted_rating"] = movies["weighted_rating"].fillna(C)

    movies["quality_score"] = movies["quality_score"].fillna(
        movies["quality_score"].median() # Movies without ratings receive a neutral quality score
    )

    return movies

def build_genre_similarity_matrix(movies: pd.DataFrame):
    """
    Build cosine similarity matrix using CountVectorizer over soup_genres
    """

    vectorizer = CountVectorizer()
    genre_matrix = vectorizer.fit_transform(movies["soup_genres"])

    similarity_genres = cosine_similarity(genre_matrix)

    return similarity_genres, vectorizer

def build_semantic_tags_similarity_matrix(
    movies: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Build cosine similarity matrix using sentence-transformer embeddings over soup_tags.
    """

    model = SentenceTransformer(model_name)

    tag_embeddings = model.encode(
        movies["soup_tags"].to_list(),
        show_progress_bar=True,
        normalize_embeddings=True
    )

    similarity_tags = cosine_similarity(tag_embeddings)

    return similarity_tags, model

def combine_similarity_matrices(
    similarity_genres,
    similarity_tags,
    weight_genres: float = 0.4,
    weight_tags: float = 0.6
):
    """
    Combine genre-based and semantic tag-based similarity matrices.
    """

    if round(weight_genres + weight_genres, 5) != 1.0:
        raise ValueError("weight_genres + weight_tags must be equal to 1.0")

    return (weight_genres * similarity_genres) + (weight_tags * similarity_tags)

def build_content_based_model(
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    ratings: pd.DataFrame,
    weight_genres: float = 0.4,
    weight_tags: float = 0.6,
    popularity_percentile: float = 0.70
):
    """
    Full pipeline for the content-based recommender.
    """
    
    movies_model = prepare_metadata_soup(movies, tags)

    movies_model = calculate_weighted_rating(
        movies_model,
        ratings,
        popularity_percentile=popularity_percentile
    )

    similarity_genres, genre_vectorizer = build_genre_similarity_matrix(movies_model)

    similarity_tags, semantic_model = build_semantic_tags_similarity_matrix(movies_model)

    similarity_content = combine_similarity_matrices(
        similarity_genres,
        similarity_tags,
        weight_genres=weight_genres,
        weight_tags=weight_tags
    )

    artifacts = {
        "movies_model": movies_model,
        "similarity_genres": similarity_genres,
        "similarity_tags": similarity_tags,
        "similarity_content": similarity_content,
        "genre_vectorizer": genre_vectorizer,
        "semantic_model": semantic_model,
    }

    return artifacts

def recommend_movies(
    title: str,
    movies_model: pd.DataFrame,
    similarity_content,
    top_n: int,
    quality_weight: float = 0.2
) -> pd.DataFrame:
    """
    Recommend movies using combined content similarity and weighted rating quality
    Final score:
    ((1 - peso_calidad) * content_similarity) + (peso_calidad * score_calidad)
    """

    if not 0 <= quality_weight <= 1:
        raise ValueError("weight_quality must be between 0 and 1")

    title_matches = movies_model[
        movies_model["title"].str.lower() == title.lower()
    ]

    if title_matches.empty:
        title_matches = movies_model[
            movies_model["title"].str.lower().str.contains(
                title.lower(),
                regex = False,
                na = False
            )
        ]

    if title_matches.empty:
        raise ValueError(f"Movie title not found: {title}")
    
    movie_index = title_matches.index[0]

    content_scores = similarity_content[movie_index]

    results = movies_model.copy()
    
    results["similarity_content"] = content_scores

    results["final_score"] = (
        ((1 - quality_weight) * results["similarity_content"])
        + (quality_weight * results["score_calidad"])
    )

    results = results.drop(index=movie_index)

    recommendations = results.sort_values(
        by="final_score",
        ascending=False
    ).head(top_n)

    columns = [
        "movieId",
        "title",
        "genres",
        "decade",
        "similarity_content",
        "score_calidad",
        "final_score",
        "v",
        "R"
    ]
    
    return recommendations[columns].reset_index(drop=True)
