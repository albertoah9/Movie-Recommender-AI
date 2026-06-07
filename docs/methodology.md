# Content-Based Filtering

Content-Based Filtering recommends items that are similar to items the user already likes

In this project, each movie is represented using its title and genres. These textual features are transformed into numerical vectors using TF-IDF. Then, cosine similarity is used to calculate how similar each movie is to every other movie

The system receives a movie title as input and returns the most similar movies according to their content representation.