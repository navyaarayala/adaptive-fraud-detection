from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score
)


def evaluate():
    Path("results").mkdir(exist_ok=True)

    baseline = joblib.load(
        "models/baseline_model.joblib"
    )

    adapted = joblib.load(
        "models/adapted_model.joblib"
    )

    _, X_test, _, y_test = joblib.load(
        "models/shifted_splits.joblib"
    )

    models = {
        "Before retraining": baseline,
        "After retraining": adapted
    }

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        scores = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(
            y_test, scores
        )

        ap = average_precision_score(
            y_test, scores
        )

        plt.plot(
            recall, precision,
            label=f"{name} (AP={ap:.3f})"
        )

        print(f"\n{name}:")
        print(classification_report(
            y_test,
            model.predict(X_test),
            zero_division=0
        ))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fraud Detection Under Distribution Shift")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/precision_recall.png")
    plt.close()

    comparison = pd.read_csv(
        "results/model_comparison.csv"
    )

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        ["Baseline", "Before retraining",
         "After retraining"],
        comparison["average_precision"],
        color=["steelblue", "tomato", "seagreen"]
    )

    plt.bar_label(bars, fmt="%.3f")
    plt.ylim(0, 1.1)
    plt.ylabel("Average Precision")
    plt.title("Model Performance Under Drift")
    plt.tight_layout()
    plt.savefig("results/model_comparison.png")
    plt.close()

    drift = pd.read_csv("results/drift_report.csv")

    plt.figure(figsize=(10, 6))

    plt.barh(
        drift["feature"],
        drift["ks_statistic"],
        color=[
            "tomato" if detected else "steelblue"
            for detected in drift["drift_detected"]
        ]
    )

    plt.axvline(
        0.05, linestyle="--",
        color="black",
        label="KS statistic threshold"
    )

    plt.xlabel("KS Statistic")
    plt.title("Feature Distribution Drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/feature_drift.png")
    plt.close()

    print("\nEvaluation completed successfully!")
    print("Three visualization charts saved.")


if __name__ == "__main__":
    evaluate()