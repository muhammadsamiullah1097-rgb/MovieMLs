from pathlib import Path

from app.services.training import run_model


required_files = [
    "movies_metadata.csv",
    "ratings_small.csv",
    "keywords.csv",
    "links_small.csv",
]
data_folder = Path(__file__).resolve().parent / "data" / "raw"
missing_files = [name for name in required_files if not (data_folder / name).is_file()]

if missing_files:
    print("Required Kaggle CSV files are missing from:")
    print(data_folder)
    for name in missing_files:
        print(f"- {name}")
    print("Download The Movies Dataset from Kaggle and copy the CSV files into this folder.")
    raise SystemExit(1)

models = [
    ("classification", "rating_tier", "random_forest"),
    ("regression", "vote_average", "random_forest_regression"),
    ("text", "primary_genre", "tfidf_naive_bayes"),
]

for task, target, model in models:
    print(f"Training {task}: {model}")
    result = run_model(None, task, target, model)
    print(f"{result['model_name']} completed")
    print(f"Metrics: {result['metrics']}")
    print(f"Saved: {result['saved_model']}")
    print("-" * 40)

print("All reference models are ready.")
