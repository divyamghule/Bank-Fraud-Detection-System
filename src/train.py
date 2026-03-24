from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "amount",
    "time_since_last_txn_min",
    "merchant_risk_score",
    "user_velocity_1h",
    "device_trust_score",
    "geo_distance_km",
]


@dataclass
class TrainingMetrics:
    roc_auc: float
    pr_auc: float
    selected_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float



def generate_synthetic_data(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    amount = rng.gamma(shape=2.2, scale=110.0, size=n_samples)
    time_since_last_txn_min = rng.exponential(scale=30.0, size=n_samples)
    merchant_risk_score = rng.beta(a=2.2, b=3.0, size=n_samples)
    user_velocity_1h = rng.poisson(lam=3.2, size=n_samples)
    device_trust_score = rng.beta(a=3.5, b=1.8, size=n_samples)
    geo_distance_km = rng.exponential(scale=45.0, size=n_samples)

    fraud_score = (
        0.0055 * amount
        + 0.34 * merchant_risk_score
        + 0.20 * user_velocity_1h
        + 0.003 * geo_distance_km
        - 0.32 * device_trust_score
        - 0.0015 * time_since_last_txn_min
    )

    noise = rng.normal(0, 0.55, n_samples)
    logits = fraud_score + noise - 1.3
    fraud_probability = 1 / (1 + np.exp(-logits))
    is_fraud = (rng.uniform(0, 1, size=n_samples) < fraud_probability).astype(int)

    data = pd.DataFrame(
        {
            "amount": amount,
            "time_since_last_txn_min": time_since_last_txn_min,
            "merchant_risk_score": merchant_risk_score,
            "user_velocity_1h": user_velocity_1h,
            "device_trust_score": device_trust_score,
            "geo_distance_km": geo_distance_km,
            "is_fraud": is_fraud,
        }
    )
    return data



def select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1))

    return (
        float(thresholds[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
        float(f1[best_idx]),
    )



def train_model(output_dir: Path) -> TrainingMetrics:
    df = generate_synthetic_data()

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=280,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold, precision, recall, f1_score = select_best_threshold(y_test.to_numpy(), y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    metrics = TrainingMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        selected_threshold=threshold,
        precision_at_threshold=precision,
        recall_at_threshold=recall,
        f1_at_threshold=f1_score,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
    }

    joblib.dump(model_bundle, output_dir / "fraud_model.joblib")

    report = classification_report(y_test, y_pred, output_dict=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "summary": asdict(metrics),
                "classification_report": report,
                "fraud_rate": float(df["is_fraud"].mean()),
            },
            metrics_file,
            indent=2,
        )

    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"

    trained_metrics = train_model(models_dir)
    print("Model trained and saved to ./models/fraud_model.joblib")
    print(
        f"ROC-AUC: {trained_metrics.roc_auc:.4f} | "
        f"PR-AUC: {trained_metrics.pr_auc:.4f} | "
        f"Threshold: {trained_metrics.selected_threshold:.4f}"
    )
