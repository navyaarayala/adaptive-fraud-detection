from pathlib import Path

import joblib
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from src.features import build_features
from src.train import create_model


def detect_drift(reference, current):
    results = []

    for column in reference.columns:
        statistic, p_value = ks_2samp(
            reference[column],
            current[column]
        )

        results.append({
            "feature": column,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": (
                p_value < 0.005 and statistic > 0.05
            )
        })

    return pd.DataFrame(results)


def run_adaptation():
    Path("results").mkdir(exist_ok=True)

    baseline_model = joblib.load(
        "models/baseline_model.joblib"
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        joblib.load("models/baseline_splits.joblib")
    )

    shifted = pd.read_csv("data/shifted.csv")
    X_shift, y_shift = build_features(shifted)

    X_adapt, X_final, y_adapt, y_final = train_test_split(
        X_shift, y_shift,
        test_size=0.4,
        random_state=42,
        stratify=y_shift
    )

    drift_report = detect_drift(X_train, X_adapt)

    drift_report.to_csv(
        "results/drift_report.csv",
        index=False
    )

    print("Feature Drift Report:")
    print(drift_report.to_string(index=False))

    baseline_scores = baseline_model.predict_proba(
        X_test
    )[:, 1]

    shifted_scores = baseline_model.predict_proba(
        X_final
    )[:, 1]

    baseline_ap = average_precision_score(
        y_test, baseline_scores
    )

    shifted_ap = average_precision_score(
        y_final, shifted_scores
    )

    # Retrain using original and newly labeled data
    X_combined = pd.concat(
        [X_train, X_adapt], ignore_index=True
    )

    y_combined = pd.concat(
        [y_train, y_adapt], ignore_index=True
    )

    adapted_model = create_model(y_combined)
    adapted_model.fit(X_combined, y_combined)

    adapted_scores = adapted_model.predict_proba(
        X_final
    )[:, 1]

    adapted_ap = average_precision_score(
        y_final, adapted_scores
    )

    joblib.dump(
        adapted_model,
        "models/adapted_model.joblib"
    )

    joblib.dump(
        (X_adapt, X_final, y_adapt, y_final),
        "models/shifted_splits.joblib"
    )

    comparison = pd.DataFrame({
        "scenario": [
            "Baseline test data",
            "Shifted data before retraining",
            "Shifted data after retraining"
        ],
        "average_precision": [
            baseline_ap,
            shifted_ap,
            adapted_ap
        ]
    })

    comparison.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print("\nModel Performance Comparison:")
    print(comparison.to_string(index=False))

    print("\nDrift analysis and retraining completed!")


if __name__ == "__main__":
    run_adaptation()