"""Train and save a machine learning model for social media virality prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "social_media_data.csv"
MODEL_PATH = MODEL_DIR / "virality_model.pkl"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.png"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

FEATURE_COLUMNS = [
    "views",
    "likes",
    "comments",
    "shares",
    "like_view_ratio",
    "comment_view_ratio",
    "share_view_ratio",
    "engagement_rate",
]


def ensure_directories() -> None:
    """Create the data and model directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_sample_data(num_posts: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic social media dataset for demonstration purposes.

    This dataset is intentionally illustrative. The real-world model should be trained
    on historical post performance data from the user's own platform.
    """
    rng = np.random.default_rng(random_state)

    views = rng.integers(500, 900000, size=num_posts)
    like_rate = rng.uniform(0.02, 0.32, size=num_posts)
    comment_rate = rng.uniform(0.003, 0.08, size=num_posts)
    share_rate = rng.uniform(0.005, 0.18, size=num_posts)

    likes = np.round(views * like_rate).astype(int)
    comments = np.round(views * comment_rate).astype(int)
    shares = np.round(views * share_rate).astype(int)

    like_view_ratio = likes / views
    comment_view_ratio = comments / views
    share_view_ratio = shares / views

    engagement_rate = ((likes + comments + shares) / views) * 100

    like_component = np.clip(like_view_ratio * 180, 0, 50)
    comment_component = np.clip(comment_view_ratio * 220, 0, 30)
    share_component = np.clip(share_view_ratio * 260, 0, 35)
    engagement_component = np.clip(engagement_rate * 1.15, 0, 30)

    virality_score = np.clip(
        10 + like_component + comment_component + share_component + engagement_component + rng.normal(0, 4, size=num_posts),
        0,
        100,
    )

    dataset = pd.DataFrame(
        {
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "like_view_ratio": like_view_ratio,
            "comment_view_ratio": comment_view_ratio,
            "share_view_ratio": share_view_ratio,
            "engagement_rate": engagement_rate,
            "virality_score": virality_score,
        }
    )

    dataset["virality_score"] = dataset["virality_score"].round(2)
    return dataset


def save_dataset(dataset: pd.DataFrame) -> None:
    """Save the sample dataset to the data directory."""
    ensure_directories()
    dataset.to_csv(DATA_PATH, index=False)


def train_and_save_model() -> dict:
    """Train the RandomForest model and save it along with evaluation metrics.

    The sample dataset is intentionally regenerated each run to avoid stale training
    artifacts from earlier versions of the project. This keeps the demo model behavior
    consistent and prevents the virality score from being stuck at the same value.
    """
    ensure_directories()

    dataset = generate_sample_data()
    save_dataset(dataset)

    X = dataset[FEATURE_COLUMNS]
    y = dataset["virality_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    importance.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Feature Importance for Virality Prediction")
    ax.set_ylabel("Importance Score")
    ax.set_xlabel("Features")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PATH, dpi=150)
    plt.close(fig)

    print("Model training complete.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved dataset to: {DATA_PATH}")
    print(f"Saved feature importance chart to: {IMPORTANCE_PATH}")
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=4))

    return metrics


def main() -> None:
    """Run the model training pipeline."""
    train_and_save_model()


if __name__ == "__main__":
    main()
