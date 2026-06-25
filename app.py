import sys
sys.path.append("src")

from pathlib import Path

import pandas as pd

import streamlit as st

from content_based import build_content_based_model, recommend_movies


DATA_PATH = Path("data/raw")

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_model():
    movies = pd.read_csv(DATA_PATH / "movies.csv")
    tags = pd.read_csv(DATA_PATH / "tags.csv")
    ratings=pd.read_csv(DATA_PATH / "ratings.csv")

    artifacts = build_content_based_model(
        movies=movies,
        tags=tags,
        ratings=ratings,
        weight_genres=0.4,
        weight_tags=0.6,
        popularity_percentile=0.7
    )

    return artifacts


def main():
    st.title("Movie Recommender")

    st.text("This application recommends movies using a content-based " \
    "recommendation system based on genres, semantic tags and rating quality.")

    with st.spinner("Loading recommendation model..."):
        artifacts = load_model()
    
    movies_model = artifacts["movies_model"]
    similarity_content = artifacts["similarity_content"]

    st.sidebar.header("Recommendation Settings")

    top_n = st.sidebar.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=12
    )

    quality_weight = st.sidebar.slider(
        "Quality weight",
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.05
    )

    movies_titles = movies_model["title"].tolist()

    selected_movie = st.selectbox(
        "Select a movie",
        movies_titles
    )

    if st.button("Recommend Movies"):
        recommendations = recommend_movies(
            title=selected_movie,
            movies_model=movies_model,
            similarity_content=similarity_content,
            top_n=top_n,
            quality_weight=quality_weight
        )

        st.subheader(f"Recommendartions based on: {selected_movie}")

        for _, row in recommendations.iterrows():
            with st.container():
                st.markdown(f"### {row['title']}")
                st.write(f"**Genres:** {row['genres']}")
                st.write(f"**Decade:** {row['decade']}")
                st.write(f"**Final Similarity Score:** {row['final_score']:.3f}")
                st.write(f"**Average Rating:** {row['R']:.2f} out of 5")
                st.write(f"**Number of Ratings:** {int(row['v'])}")
                st.divider()


if __name__ == "__main__":
    main()
