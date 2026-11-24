import pandas as pd

from src.generate_data import generate_transactions
from src.features import build_features
from src.drift import detect_drift


def test_dataset_size():
    df = generate_transactions(1000)
    assert len(df) == 1000


def test_binary_labels():
    df = generate_transactions(1000)
    assert set(df["is_fraud"].unique()) == {0, 1}


def test_feature_dimensions():
    df = generate_transactions(100)
    X, y = build_features(df)
    assert X.shape == (100, 10)
    assert len(y) == 100


def test_no_missing_values():
    df = generate_transactions(100)
    X, y = build_features(df)
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_no_label_leakage():
    df = generate_transactions(100)
    X, y = build_features(df)
    assert "is_fraud" not in X.columns


def test_reproducibility():
    first = generate_transactions(100, seed=42)
    second = generate_transactions(100, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_drift_detection():
    reference = pd.DataFrame({
        "feature": range(1000)
    })

    current = pd.DataFrame({
        "feature": range(1000, 2000)
    })

    report = detect_drift(reference, current)
    assert bool(report.loc[0, "drift_detected"])


def test_no_drift():
    reference = pd.DataFrame({
        "feature": range(1000)
    })

    report = detect_drift(reference, reference)
    assert not bool(report.loc[0, "drift_detected"])