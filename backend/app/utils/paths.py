from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = APP_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
DERIVED_DIR = DATA_DIR / "derived"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"

for folder in [UPLOAD_DIR, DERIVED_DIR, ARTIFACT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
