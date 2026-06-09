# Data Documentation

## Dataset Source
This project uses the MovieLens Latest Small dataset provided by GroupLens Research at the University of Minnesota
Dataset URL: https://grouplens.org/datasets/movielens/latest/

## Dataset Files

### movies.csv
Contains information related to movies

| Column | Description |
| -------- | -------- |
| movieid | Unique movie identifier |
| title | Movie title and release year |
| genres | Movie genres separated by \||

Example: 
| movieid | title | genres |
| ------- | ----- | ------ |
| 1 | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy|

### ratings.csv
Contains ratings provided by users
| Column | Description |
| ------ | ----------- |
| userId | Unique user identifier |
| movieId | Movie identifier |
| rating | Rating from 0.5 to 5.0 |
| timestamp | Unix timestamp of the rating |

Example: 
| userId | movieId | rating | timestamp |
| ------ | ------- | ------ | --------- |
| 1 | 1 | 4.0 | 964982703 |

## Data Usage

### Content-Based Recommender

The content-based recommendation system uses:

* Movie titles
* Genres
* Additional metadata (when available)

to compute similarities between movies.

### Collaborative Filtering

The collaborative filtering model uses:

* User ratings
* User-item interactions

to learn user preferences and generate personalized recommendations.

## Data Version
Dataset: MovieLens Latest Small (movies.csv & ratings.csv)
Downloaded on: 31/05/2026

Dataset: MovieLens Latest Small (tags.csv)
Downloaded on: 7/06/2026
