from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_content_features(movies):
    movies = movies.copy()

    movies["content"] = (
        movies["clean_title"].fillna("") + " " + 
        movies["genres"].fillna("").str.replace("|", " ", regex=False)
    )

    return movies

def build_tfidf_matrix(movies):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["content"])

    return tfidf_matrix, vectorizer

def compute_similarity_matrix(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)

def recommend_movies(movie_title, movies, similarity_matrix, top_n=10):
    indices = movies.reset_index().set_index("clean_title")["index"]

    if movie_title not in indices:
        return None
    
    movie_index = indices[movie_title]
    
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n+1] # movie with index 0 is the movie itself

    movie_indices = [i[0] for i in similarity_scores]

    recommendations = movies.iloc[movie_indices][
        ["movieId", "title", "genres", "year"]
    ].copy()

    recommendations["similarity_score"] = [
        round(score, 3) for _, score in similarity_scores
    ]

    return recommendations
