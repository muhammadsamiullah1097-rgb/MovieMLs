import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline

from app.evaluation.metrics import classification_metrics, regression_metrics
from app.preprocessing.imbalance import make_smote
from app.preprocessing.cleaning import clean_data
from app.preprocessing.dimensionality_reduction import run_pca
from app.preprocessing.pipeline import make_preprocessor
from app.services.model_registry import CLASSIFIERS, REGRESSORS, TEXT_MODELS
from app.services.movie_data import get_tabular_data, get_text_data, MOVIE_FEATURES, POLYNOMIAL_FEATURES
from app.utils.paths import ARTIFACT_DIR

MAX_ROWS = 8500
RANDOM_STATE = 42


def make_dense(data):
    return data.toarray() if sparse.issparse(data) else data


def sample_data(data, limit=MAX_ROWS):
    if len(data) > limit:
        return data.sample(limit, random_state=RANDOM_STATE).reset_index(drop=True)
    return data.reset_index(drop=True)


def _save_model(model, task, model_key, target, features):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{task}_{model_key}.joblib"
    joblib.dump({"model": model, "task": task, "target": target, "features": features}, path, compress=3)
    return path.name


def _fit_tabular(data, task, target, model_key, test_size, scaler, use_smote, save=True):
    frame = clean_data(get_tabular_data(task))
    frame = sample_data(frame)
    if task == "classification":
        if target not in {"rating_tier", "vote_average"}:
            raise ValueError("Movie classification predicts the rating_tier created from vote_average.")
        target = "rating_tier"
        model_map = CLASSIFIERS
        y = frame[target].astype(str)
        x = frame[MOVIE_FEATURES].copy()
    else:
        if target not in {"vote_average", "rating_tier"}:
            raise ValueError("Movie regression predicts vote_average.")
        target = "vote_average"
        model_map = REGRESSORS
        y = frame[target].astype(float)
        x = frame[MOVIE_FEATURES].copy()

    if model_key not in model_map:
        raise ValueError("Select a valid model for this task.")

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=float(test_size), random_state=RANDOM_STATE, stratify=stratify
    )

    if task == "regression" and model_key == "linear_regression":
        features = ["runtime"]
        preprocessor = make_preprocessor(x_train[features], scaler)
        estimator = model_map[model_key][1]()
        model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    elif task == "regression" and model_key == "polynomial_regression":
        features = POLYNOMIAL_FEATURES
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("polynomial_model", model_map[model_key][1]()),
        ])
        x_train = x_train[features]
        x_test = x_test[features]
    else:
        features = MOVIE_FEATURES
        preprocessor = make_preprocessor(x_train, scaler)
        estimator = model_map[model_key][1]()
        steps = [("preprocess", preprocessor)]
        apply_smote = task == "classification" and use_smote and model_key != "dummy_baseline"
        if apply_smote:
            steps.append(("smote", make_smote()))
        if task == "classification" and model_key == "naive_bayes":
            steps.append(("dense", FunctionTransformer(make_dense, accept_sparse=True)))
        steps.append(("model", estimator))
        model_class = ImbPipeline if apply_smote else Pipeline
        model = model_class(steps)

    started = time.perf_counter()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    elapsed = round(time.perf_counter() - started, 3)

    if task == "classification":
        score = None
        final_estimator = model.named_steps.get("model")
        if hasattr(final_estimator, "predict_proba"):
            try:
                score = model.predict_proba(x_test)
            except Exception:
                score = None
        metrics = classification_metrics(y_test, predictions, score)
        preview = pd.DataFrame({"Actual": y_test.astype(str).to_numpy(), "Predicted": predictions.astype(str)})
        metric_name = "Weighted F1"
        metric_value = metrics["f1"]
    else:
        metrics = regression_metrics(y_test, predictions)
        preview = pd.DataFrame({"Actual rating": y_test.to_numpy(), "Predicted rating": predictions})
        preview = preview.round(2)
        metric_name = "R-squared"
        metric_value = metrics["r2"]

    artifact = _save_model(model, task, model_key, target, features) if save else None
    return {
        "task": task,
        "model": model_key,
        "model_name": model_map[model_key][0],
        "target": target,
        "smote_applied": bool(task == "classification" and use_smote and model_key != "dummy_baseline"),
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": features,
        "metrics": metrics,
        "preview": preview.head(15).replace({np.nan: None}).to_dict(orient="records"),
        "labels": sorted(y.astype(str).unique().tolist()) if task == "classification" else [],
        "metric_name": metric_name,
        "metric_value": metric_value,
        "seconds": elapsed,
        "saved_model": artifact,
        "note": "Results are measured on a held-out test split. They are not guaranteed to be perfect on new movies.",
    }


def _fit_text(data, task, model_key, test_size, save=True):
    frame = sample_data(get_text_data(), limit=12000)
    x = (frame["title"].astype(str) + " " + frame["overview"].astype(str))
    y = frame["primary_genre"].astype(str)
    if model_key not in TEXT_MODELS:
        raise ValueError("Select a valid text classification model.")
    counts = y.value_counts()
    valid = y.isin(counts[counts >= 2].index)
    x, y = x[valid], y[valid]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=float(test_size), random_state=RANDOM_STATE, stratify=y
    )
    model = TEXT_MODELS[model_key][1]()
    started = time.perf_counter()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    elapsed = round(time.perf_counter() - started, 3)
    metrics = classification_metrics(y_test, predictions)
    preview = pd.DataFrame({"Actual genre": y_test.to_numpy(), "Predicted genre": predictions})
    artifact = _save_model(model, "text", model_key, "primary_genre", ["overview"]) if save else None
    return {
        "task": "text",
        "model": model_key,
        "model_name": TEXT_MODELS[model_key][0],
        "target": "primary_genre",
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": ["title", "overview"],
        "metrics": metrics,
        "preview": preview.head(15).to_dict(orient="records"),
        "labels": sorted(y.unique().tolist()),
        "metric_name": "Weighted F1",
        "metric_value": metrics["f1"],
        "seconds": elapsed,
        "saved_model": artifact,
        "note": "The model predicts one of the eight most common primary genres from the movie title and overview. The training set is balanced by genre.",
    }


def run_model(data, task, target, model, test_size=0.2, scaler="standard", use_smote=False):
    if task == "text":
        return _fit_text(data, task, model, test_size)
    if task not in {"classification", "regression"}:
        raise ValueError("Choose classification, regression or text classification.")
    return _fit_tabular(data, task, target, model, test_size, scaler, use_smote)


def compare_models(data, task, target, test_size=0.2, scaler="standard", use_smote=False):
    if task == "text":
        models = TEXT_MODELS
        runner = lambda key: _fit_text(data, task, key, test_size, save=False)
    elif task == "classification":
        models = CLASSIFIERS
        runner = lambda key: _fit_tabular(data, task, target, key, test_size, scaler, use_smote, save=False)
    elif task == "regression":
        models = REGRESSORS
        runner = lambda key: _fit_tabular(data, task, target, key, test_size, scaler, False, save=False)
    else:
        raise ValueError("Choose a valid task before comparing models.")

    results = []
    errors = []
    for key in models:
        try:
            result = runner(key)
            results.append({
                "model": key,
                "model_name": result["model_name"],
                "metric_name": result["metric_name"],
                "score": result["metric_value"],
                "seconds": result["seconds"],
            })
        except Exception as exc:
            errors.append({"model": key, "error": str(exc)[:180]})
    if not results:
        raise ValueError("No models completed. " + " | ".join(item["error"] for item in errors))
    results.sort(key=lambda item: item["score"], reverse=True)
    best = results[0]
    return {
        "task": task,
        "metric": "Weighted F1" if task != "regression" else "R-squared",
        "results": results,
        "best_model": best,
        "errors": errors,
        "note": "All models use the same random seed and held-out split settings. Compare test metrics rather than training scores.",
    }


def feature_report(data=None, target="rating_tier"):
    from app.preprocessing.feature_selection import select_features
    from sklearn.impute import SimpleImputer

    frame = sample_data(get_tabular_data("classification"), limit=3500)
    x = frame[MOVIE_FEATURES]
    y = frame["rating_tier"].astype(str)
    selected, rankings = select_features(x, y, count=8)
    numeric = x[["runtime", "release_year", "budget_log", "revenue_log", "popularity_log", "vote_count_log"]]
    matrix = SimpleImputer(strategy="median").fit_transform(numeric)
    pca_result = run_pca(matrix, 2)
    return {
        "selected_features": selected,
        "feature_count": len(selected),
        "rfe_rankings": [int(value) for value in rankings[:40]],
        "pca": pca_result,
    }
