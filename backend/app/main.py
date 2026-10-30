from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.movie_data import dataset_summary, get_tabular_data, preprocessing_report, get_text_data
from app.services.training import run_model, compare_models, feature_report
from app.services.model_registry import CLASSIFIERS, REGRESSORS, TEXT_MODELS
from app.services.recommendation import content_recommendations, collaborative_recommendations, search_movies
from app.utils.paths import BACKEND_DIR, ARTIFACT_DIR

app = FastAPI(title="MovieLens ML Studio", version="2.0.0")
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class RunRequest(BaseModel):
    task: str = "classification"
    target: str = "rating_tier"
    model: str = "random_forest"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    scaler: str = "standard"
    use_smote: bool = False


class CompareRequest(BaseModel):
    task: str = "classification"
    target: str = "rating_tier"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    scaler: str = "standard"
    use_smote: bool = False


class ContentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    limit: int = Field(default=10, ge=3, le=20)


class CollaborativeRequest(BaseModel):
    user_id: int = Field(ge=1, le=10000)
    limit: int = Field(default=10, ge=3, le=20)


@app.get("/", response_class=HTMLResponse)
def home():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "MovieLens ML Studio"}


@app.get("/api/datasets")
def datasets():
    summary = dataset_summary()
    return {
        "id": "the_movies_dataset",
        "name": "The Movies Dataset (Kaggle)",
        "movie_rows": summary["movie_rows"],
        "rating_rows": summary["rating_rows"],
        "files": summary["files"],
    }


@app.get("/api/summary")
def summary():
    return dataset_summary()


@app.get("/api/models")
def models():
    return {
        "classification": [{"id": key, "name": name} for key, (name, _) in CLASSIFIERS.items()],
        "regression": [{"id": key, "name": name} for key, (name, _) in REGRESSORS.items()],
        "text": [{"id": key, "name": name} for key, (name, _) in TEXT_MODELS.items()],
    }


@app.get("/api/metadata/{task}")
def metadata(task: str):
    try:
        if task == "text":
            frame = get_text_data()
            return {
                "rows": int(len(frame)),
                "columns": ["title", "overview", "primary_genre"],
                "target": "primary_genre",
                "features": ["title", "overview"],
            }
        frame = get_tabular_data(task)
        target = "rating_tier" if task == "classification" else "vote_average"
        return {
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
            "target": target,
            "features": [column for column in frame.columns if column != target],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/run")
def run(request: RunRequest):
    try:
        return run_model(None, request.task, request.target, request.model, request.test_size, request.scaler, request.use_smote)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/compare")
def compare(request: CompareRequest):
    try:
        return compare_models(None, request.task, request.target, request.test_size, request.scaler, request.use_smote)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/preprocessing")
def preprocessing():
    try:
        return preprocessing_report()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/features")
def features():
    try:
        return feature_report()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/movies/search")
def movie_search(q: str, limit: int = 8):
    try:
        return search_movies(q, max(1, min(limit, 15)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/recommend/content")
def recommend_content(request: ContentRequest):
    try:
        return content_recommendations(request.title, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/recommend/collaborative")
def recommend_collaborative(request: CollaborativeRequest):
    try:
        return collaborative_recommendations(request.user_id, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/artifacts")
def artifacts():
    files = sorted(ARTIFACT_DIR.glob("*.joblib"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [{"name": path.name, "size_kb": round(path.stat().st_size / 1024, 1)} for path in files]
