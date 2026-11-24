from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score
)

from src.features import build_features


def create_model(y_train):
    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()

    return XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negatives / positives,
        eval_metric="logloss",
        random_state=42,
        n_jobs=2
    )


def train_model():
    df = pd.read_csv("data/baseline.csv")

    X, y = build_features(df)

    X_train, X_holdout, y_train, y_holdout = (
        train_test_split(
            X, y,
            test_size=0.4,
            random_state=42,
            stratify=y
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_holdout, y_holdout,
            test_size=0.5,
            random_state=42,
            stratify=y_holdout
        )
    )

    model = create_model(y_train)
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        model,
        "models/baseline_model.joblib"
    )

    joblib.dump(
        (X_train, X_val, X_test,
         y_train, y_val, y_test),
        "models/baseline_splits.joblib"
    )

    predictions = model.predict(X_val)
    scores = model.predict_proba(X_val)[:, 1]

    print("XGBoost model trained successfully!")

    print("\nValidation Results:")
    print(classification_report(
        y_val, predictions, zero_division=0
    ))

    print("Average Precision:",
          average_precision_score(y_val, scores))

    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))


if __name__ == "__main__":
    train_model()