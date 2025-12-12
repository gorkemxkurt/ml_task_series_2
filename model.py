from __future__ import annotations

from pathlib import Path
import joblib

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def train_and_save_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X, y = iris.data, iris.target

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)


def load_model():
    if not MODEL_PATH.exists():
        train_and_save_model()
    return joblib.load(MODEL_PATH)
