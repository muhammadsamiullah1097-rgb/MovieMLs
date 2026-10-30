# Dataset and target design

Source: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

## CSV files used

- `movies_metadata.csv`: tabular prediction, text classification and movie search.
- `ratings_small.csv`: user-item ratings for collaborative recommendations.
- `keywords.csv`: tag words for content similarity.
- `links_small.csv`: ID mapping between MovieLens and TMDB.

## Target definitions

| ML task | Target | Inputs |
|---|---|---|
| Classification | `rating_tier` | Runtime, release year, transformed budget/revenue/popularity/vote count, original language, primary genre and adult flag |
| Regression | `vote_average` | Same movie metadata inputs; vote_average is removed from the input features |
| Text classification | `primary_genre` (eight most common genres) | Movie title + `overview` free text |

The classification label is created using fixed thresholds: `<5` Low, `5 to <7` Medium, `7 to 10` High. Data with missing/zero vote average or fewer than 10 votes is excluded from classification and regression.

## Data quality notes

The source CSV contains invalid/non-numeric movie IDs, duplicate IDs, missing metadata and zero placeholders in some numeric fields. The loader converts IDs and numeric fields safely, removes rows without a usable movie ID/title, removes duplicate TMDB IDs, and treats zero budget/revenue as missing before taking log values.

The full `ratings.csv` and `credits.csv` from the original archive are not used. The project uses `ratings_small.csv`, which is enough for the included 671-user collaborative filtering demonstration, and avoids including the much larger rating file.
