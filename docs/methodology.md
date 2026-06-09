# Methodology

## Content-Based Recommendation System

This project implements an advanced content-based recommendation system using multiple sources of movie information.

The model does not rely only on movie genres because genre-based similarity can produce many identical similarity scores. It also avoids using movie titles as a main feature because titles often introduce noise and do not necessarily describe the movie content.

## Metadata Soup

Each movie is represented using a metadata soup composed of:

- Cleaned genres
- Release decade
- User-generated tags

The release year is extracted from the movie title and transformed into a decade label, such as `decada_1990s`.

## Genre-Based Similarity

Genres and decade information are represented using `CountVectorizer`. Cosine similarity is then used to calculate similarity between movies based on structured metadata.

## Semantic Tag Similarity

User-generated tags are transformed into dense semantic embeddings using the `all-MiniLM-L6-v2` sentence-transformer model.

This allows the system to capture semantic relationships between tags instead of relying only on exact word matches.

## Weighted Rating Score

To reduce low-quality recommendations, the system calculates an IMDb-style weighted rating:

WR = (v / (v + m)) * R + (m / (v + m)) * C

Where:

- `v` is the number of ratings for the movie.
- `R` is the average rating for the movie.
- `C` is the global average rating.
- `m` is the 70th percentile of the number of votes.

The weighted rating is normalized to a value between 0 and 1 using MinMaxScaler and stored as `score_calidad`.

## Final Recommendation Score

The final recommendation score combines content similarity and quality:

Final Score = ((1 - peso_calidad) * Content Similarity) + (peso_calidad * Quality Score)

This approach balances similarity with movie quality and popularity.