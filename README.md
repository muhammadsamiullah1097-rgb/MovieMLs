# MovieML Studio — ML Midterm Project

A student-level Machine Learning project based on **The Movies Dataset** from Kaggle. It includes a FastAPI backend, a browser-based website, selectable ML models, model comparison, preprocessing, feature analysis, and movie recommendations.

## Dataset included

The following four original Kaggle CSV files are already included in `backend/data/raw/`:

- `movies_metadata.csv`
- `ratings_small.csv`
- `keywords.csv`
- `links_small.csv`

Kaggle source: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

## ML modules

- **Classification:** 12 models for movie rating-tier classification.
- **Regression:** 8 models for predicting `vote_average`.
- **Text classification:** 2 models for primary-genre prediction from movie title and overview.
- **Preprocessing:** missing-value handling, duplicate checks, encoding, scaling, SMOTE, RFE, and PCA.
- **Recommendations:** content-based and collaborative filtering.

## Run locally on Windows

Use the commands below in PowerShell. No `.bat` file is required.

### 1. Open the backend folder

Open this project folder in VS Code, open the terminal, and run:

```powershell
cd backend
pip install -r requirements.txt
python train_models.py
python -m uvicorn app.main:app --reload
```

### 5. Open the website

Open this address in Chrome:

**http://127.0.0.1:8000**

API documentation:

**http://127.0.0.1:8000/docs**

Keep the terminal open while using the website. Press `Ctrl + C` to stop the server.

## Train the reference models again (optional)

Only do this if you intentionally want to retrain the three reference models or have deleted the saved model files. The Kaggle CSVs are already included in `backend/data/raw/`.

From the `backend` folder, run:

```powershell
python train_models.py
```

This trains one reference model for each task: classification, regression, and text classification. The website can run the other selectable algorithms directly when you choose them.

## Project structure

```text
MovieML_Midterm_Project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── evaluation/
│   │   ├── models/
│   │   ├── preprocessing/
│   │   ├── services/
│   │   └── utils/
│   ├── artifacts/
│   ├── data/
│   │   ├── raw/
│   │   ├── derived/
│   │   └── uploads/
│   ├── requirements.txt
│   └── train_models.py
├── frontend/
├── docs/
├── tests/
└── README.md
```

## Main API routes

| Route | Purpose |
|---|---|
| `GET /api/health` | Check API status |
| `GET /api/datasets` | Dataset details |
| `GET /api/summary` | CSV record counts |
| `GET /api/models` | List available models |
| `POST /api/run` | Run one selected model |
| `POST /api/compare` | Compare models for a task |
| `GET /api/preprocessing` | Data quality report |
| `GET /api/features` | RFE and PCA report |
| `POST /api/recommend/content` | Content-based recommendations |
| `POST /api/recommend/collaborative` | Collaborative recommendations |

## Notes

- Saved reference models were created with scikit-learn 1.8.0. If scikit-learn shows a version warning, the app can still train a selected model from the included CSVs.
- Evaluation uses a held-out test split. Scores are not guaranteed to be perfect.
- Classification groups are derived from TMDB `vote_average`; this is not a personal rating prediction.
- Collaborative recommendations use the users and ratings in `ratings_small.csv`.
