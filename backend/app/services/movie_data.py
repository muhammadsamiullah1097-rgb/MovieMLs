import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.paths import RAW_DIR

MOVIE_FEATURES = [
    "release_year", "runtime", "budget_log", "revenue_log",
    "popularity_log", "vote_count_log", "original_language", "primary_genre", "adult",
]
NUMERIC_FEATURES = [
    "release_year", "runtime", "budget_log", "revenue_log",
    "popularity_log", "vote_count_log",
]
POLYNOMIAL_FEATURES = ["runtime", "release_year", "budget_log", "revenue_log"]


def parse_names(value):
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        return [str(item.get("name", "")).strip() for item in items if isinstance(item, dict) and item.get("name")]
    except (ValueError, SyntaxError, TypeError):
        return []


@lru_cache(maxsize=1)
def get_movies():
    path = RAW_DIR / "movies_metadata.csv"
    data = pd.read_csv(path, low_memory=False)
    data["tmdb_id"] = pd.to_numeric(data["id"], errors="coerce")
    data = data.dropna(subset=["tmdb_id", "title"]).copy()
    data["tmdb_id"] = data["tmdb_id"].astype("int64")
    data = data.drop_duplicates("tmdb_id", keep="first").reset_index(drop=True)

    data["genres_list"] = data["genres"].apply(parse_names)
    data["primary_genre"] = data["genres_list"].apply(lambda values: values[0] if values else "Unknown")
    data["genres_text"] = data["genres_list"].apply(lambda values: " ".join(values))
    data["release_year"] = pd.to_datetime(data["release_date"], errors="coerce").dt.year

    for column in ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["budget_log"] = np.log1p(data["budget"].where(data["budget"] > 0))
    data["revenue_log"] = np.log1p(data["revenue"].where(data["revenue"] > 0))
    data["popularity_log"] = np.log1p(data["popularity"].where(data["popularity"] >= 0))
    data["vote_count_log"] = np.log1p(data["vote_count"].where(data["vote_count"] >= 0))
    data["original_language"] = data["original_language"].fillna("unknown").astype(str)
    data["adult"] = data["adult"].fillna("unknown").astype(str)
    data["overview"] = data["overview"].fillna("").astype(str)
    data["title"] = data["title"].astype(str)
    data["runtime"] = data["runtime"].where(data["runtime"] > 0)
    return data


def get_tabular_data(task):
    data = get_movies().copy()
    data = data[(data["vote_average"] > 0) & (data["vote_count"] >= 10)].copy()
    data["rating_tier"] = pd.cut(
        data["vote_average"],
        bins=[0, 5, 7, 10],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    data = data.dropna(subset=["rating_tier", "vote_average"])
    if task == "classification":
        return data[MOVIE_FEATURES + ["rating_tier"]].reset_index(drop=True)
    if task == "regression":
        return data[MOVIE_FEATURES + ["vote_average"]].reset_index(drop=True)
    raise ValueError("Choose classification or regression for this dataset.")


def get_text_data():
    data = get_movies()[["title", "overview", "primary_genre"]].copy()
    data["overview"] = data["overview"].str.strip()
    data = data[(data["overview"].str.len() >= 30) & (data["primary_genre"] != "Unknown")]
    common = data["primary_genre"].value_counts().head(8).index
    data = data[data["primary_genre"].isin(common)]
    per_genre = min(1400, int(data["primary_genre"].value_counts().min()))
    data = data.groupby("primary_genre", group_keys=False).sample(n=per_genre, random_state=42)
    return data.reset_index(drop=True)


def dataset_summary():
    movies = get_movies()
    ratings_path = RAW_DIR / "ratings_small.csv"
    ratings = pd.read_csv(ratings_path)
    keywords = pd.read_csv(RAW_DIR / "keywords.csv", usecols=["id"])
    return {
        "movie_rows": int(len(movies)),
        "rating_rows": int(len(ratings)),
        "rating_users": int(ratings["userId"].nunique()),
        "rating_movies": int(ratings["movieId"].nunique()),
        "keyword_rows": int(len(keywords)),
        "genres": sorted([item for item in movies["primary_genre"].unique() if item != "Unknown"]),
        "files": [
            {"name": "movies_metadata.csv", "purpose": "Movie details, genres and TMDB ratings", "size_mb": round((RAW_DIR / "movies_metadata.csv").stat().st_size / 1048576, 2)},
            {"name": "ratings_small.csv", "purpose": "MovieLens user ratings for collaborative recommendations", "size_mb": round((RAW_DIR / "ratings_small.csv").stat().st_size / 1048576, 2)},
            {"name": "keywords.csv", "purpose": "Movie keywords for content recommendations", "size_mb": round((RAW_DIR / "keywords.csv").stat().st_size / 1048576, 2)},
            {"name": "links_small.csv", "purpose": "Maps MovieLens movie IDs to TMDB movie IDs", "size_mb": round((RAW_DIR / "links_small.csv").stat().st_size / 1048576, 2)},
        ],
    }


def preprocessing_report():
    movies = get_movies()
    feature_data = get_tabular_data("classification")
    raw_ids = pd.read_csv(RAW_DIR / "movies_metadata.csv", usecols=["id"])
    duplicate_ids = int(raw_ids["id"].duplicated().sum())
    missing = movies[["runtime", "release_year", "budget", "revenue", "popularity", "vote_average", "vote_count", "original_language"]].isna().sum()
    return {
        "rows": int(len(movies)),
        "columns": int(len(movies.columns)),
        "duplicate_rows": duplicate_ids,
        "model_rows": int(len(feature_data)),
        "missing": [{"column": key, "count": int(value)} for key, value in missing.items()],
        "preview_columns": ["title", "release_year", "primary_genre", "runtime", "vote_average", "vote_count"],
        "preview": movies[["title", "release_year", "primary_genre", "runtime", "vote_average", "vote_count"]].head(8).replace({np.nan: None}).to_dict(orient="records"),
    }
