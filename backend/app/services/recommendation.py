import ast
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from app.services.movie_data import get_movies
from app.utils.paths import RAW_DIR


def keyword_names(value):
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        return " ".join(str(item.get("name", "")) for item in items if isinstance(item, dict))
    except (ValueError, SyntaxError, TypeError):
        return ""


@lru_cache(maxsize=1)
def content_index():
    movies = get_movies().copy()
    keywords = pd.read_csv(RAW_DIR / "keywords.csv")
    keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce")
    keywords = keywords.dropna(subset=["id"]).drop_duplicates("id")
    keywords["keyword_text"] = keywords["keywords"].apply(keyword_names)
    movies = movies.merge(keywords[["id", "keyword_text"]], left_on="tmdb_id", right_on="id", how="left")
    movies["content_text"] = (
        movies["title"].fillna("") + " " + movies["overview"].fillna("") + " " +
        movies["genres_text"].fillna("") + " " + movies["keyword_text"].fillna("")
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=35000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    matrix = vectorizer.fit_transform(movies["content_text"])
    searcher = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
    searcher.fit(matrix)
    return movies.reset_index(drop=True), matrix, searcher


def find_movie(query):
    movies, _, _ = content_index()
    text = str(query).strip().lower()
    if not text:
        raise ValueError("Enter a movie title first.")
    exact = movies[movies["title"].str.lower() == text]
    if not exact.empty:
        return int(exact.index[0])
    matches = movies[movies["title"].str.lower().str.contains(text, regex=False, na=False)]
    if not matches.empty:
        return int(matches.index[0])
    raise ValueError("Movie title was not found. Try a shorter or more exact title.")


def content_recommendations(title, limit=10):
    movies, matrix, searcher = content_index()
    index = find_movie(title)
    count = max(2, min(int(limit) + 1, 21))
    distances, indices = searcher.kneighbors(matrix[index], n_neighbors=count)
    output = []
    for distance, movie_index in zip(distances[0], indices[0]):
        if movie_index == index:
            continue
        row = movies.iloc[int(movie_index)]
        output.append({
            "title": str(row["title"]),
            "year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "genre": str(row["primary_genre"]),
            "similarity": round(float(1 - distance), 3),
            "overview": str(row["overview"])[:240],
        })
        if len(output) >= limit:
            break
    source = movies.iloc[index]
    return {
        "method": "Content-based TF-IDF",
        "selected_movie": str(source["title"]),
        "recommendations": output,
        "note": "Recommendations compare movie descriptions, genres and Kaggle keyword tags.",
    }


@lru_cache(maxsize=1)
def collaborative_index():
    ratings = pd.read_csv(RAW_DIR / "ratings_small.csv")
    links = pd.read_csv(RAW_DIR / "links_small.csv")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)
    ratings = ratings.merge(links[["movieId", "tmdbId"]], on="movieId", how="inner")

    movie_ids = np.sort(ratings["tmdbId"].unique())
    user_ids = np.sort(ratings["userId"].unique())
    movie_map = {value: index for index, value in enumerate(movie_ids)}
    user_map = {value: index for index, value in enumerate(user_ids)}
    row = ratings["tmdbId"].map(movie_map).to_numpy()
    col = ratings["userId"].map(user_map).to_numpy()
    values = ratings["rating"].to_numpy(dtype=np.float32)
    matrix = csr_matrix((values, (row, col)), shape=(len(movie_ids), len(user_ids)))
    searcher = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
    searcher.fit(matrix)
    return ratings, movie_ids, user_ids, movie_map, matrix, searcher


def collaborative_recommendations(user_id, limit=10):
    ratings, movie_ids, user_ids, movie_map, matrix, searcher = collaborative_index()
    user_id = int(user_id)
    if user_id not in set(user_ids.tolist()):
        raise ValueError(f"User ID must be one of the MovieLens users (1 to {int(user_ids.max())}).")

    user_ratings = ratings[ratings["userId"] == user_id].sort_values("rating", ascending=False)
    favorite_ratings = user_ratings[user_ratings["rating"] >= 3.5].head(30)
    if favorite_ratings.empty:
        favorite_ratings = user_ratings.head(15)

    scores = {}
    support = {}
    for row in favorite_ratings.itertuples():
        source_index = movie_map.get(int(row.tmdbId))
        if source_index is None:
            continue
        distances, indices = searcher.kneighbors(matrix[source_index], n_neighbors=min(26, len(movie_ids)))
        preference = max(float(row.rating) - 2.5, 0.25)
        for distance, candidate_index in zip(distances[0], indices[0]):
            candidate_id = int(movie_ids[int(candidate_index)])
            if candidate_id == int(row.tmdbId):
                continue
            similarity = max(0.0, 1.0 - float(distance))
            scores[candidate_id] = scores.get(candidate_id, 0.0) + similarity * preference
            support[candidate_id] = support.get(candidate_id, 0) + 1

    watched = set(user_ratings["tmdbId"].astype(int).tolist())
    ranked = sorted(
        ((movie_id, score / (1 + 0.08 * (support[movie_id] - 1))) for movie_id, score in scores.items() if movie_id not in watched),
        key=lambda item: item[1], reverse=True,
    )
    movies = get_movies().set_index("tmdb_id", drop=False)
    output = []
    for tmdb_id, score in ranked:
        if tmdb_id not in movies.index:
            continue
        row = movies.loc[tmdb_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        output.append({
            "title": str(row["title"]),
            "year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "genre": str(row["primary_genre"]),
            "score": round(float(score), 3),
            "overview": str(row["overview"])[:240],
        })
        if len(output) >= limit:
            break
    return {
        "method": "Collaborative filtering with item similarity",
        "user_id": user_id,
        "based_on_ratings": int(len(user_ratings)),
        "recommendations": output,
        "note": "Suggestions use rating patterns from similar MovieLens users and exclude movies already rated by this user.",
    }


def search_movies(query, limit=8):
    movies = get_movies()
    text = str(query).strip().lower()
    if not text:
        return []
    matches = movies[movies["title"].str.lower().str.contains(text, regex=False, na=False)].head(limit)
    return [
        {"title": row.title, "year": int(row.release_year) if pd.notna(row.release_year) else None}
        for row in matches.itertuples()
    ]
