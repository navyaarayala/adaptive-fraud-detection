import numpy as np
import pandas as pd


def build_features(df):
    features = df.copy()

    features["log_amount"] = np.log1p(
        features["transaction_amount"]
    )

    features["velocity_risk"] = (
        features["transactions_last_hour"]
        * features["device_risk_score"]
    )

    features["amount_velocity"] = (
        features["log_amount"]
        * features["transactions_last_hour"]
    )

    features["is_night_transaction"] = (
        features["transaction_hour"] < 6
    ).astype(int)

    features["is_new_account"] = (
        features["account_age_days"] < 30
    ).astype(int)

    feature_columns = [
        "transaction_amount",
        "transactions_last_hour",
        "device_risk_score",
        "transaction_hour",
        "account_age_days",
        "log_amount",
        "velocity_risk",
        "amount_velocity",
        "is_night_transaction",
        "is_new_account"
    ]

    X = features[feature_columns]
    y = features["is_fraud"]

    return X, y


if __name__ == "__main__":
    baseline = pd.read_csv("data/baseline.csv")
    shifted = pd.read_csv("data/shifted.csv")

    X_base, y_base = build_features(baseline)
    X_shift, y_shift = build_features(shifted)

    print("Feature engineering completed!")
    print("Baseline shape:", X_base.shape)
    print("Shifted shape:", X_shift.shape)
    print("Baseline missing values:",
          X_base.isnull().sum().sum())
    print("Shifted missing values:",
          X_shift.isnull().sum().sum())