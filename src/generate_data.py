from pathlib import Path

import numpy as np
import pandas as pd


def generate_transactions(n_samples=10000, seed=42, drift=False):
    rng = np.random.default_rng(seed)

    fraud = rng.binomial(1, 0.10, n_samples)

    amount = rng.lognormal(
        mean=np.where(fraud == 1, 5.5, 4.0),
        sigma=1.0
    )

    velocity = rng.poisson(
        np.where(fraud == 1, 8, 2)
    )

    device_risk = rng.beta(
        np.where(fraud == 1, 5, 2),
        np.where(fraud == 1, 2, 5)
    )

    # Simulate changing fraud behavior
    if drift:
        mask = fraud == 1

        amount[mask] *= 0.5

        velocity[mask] = rng.poisson(
            4, mask.sum()
        )

        device_risk[mask] = rng.beta(
            3, 3, mask.sum()
        )

    return pd.DataFrame({
        "transaction_amount": amount.round(2),
        "transactions_last_hour": velocity,
        "device_risk_score": device_risk,
        "transaction_hour": rng.integers(
            0, 24, n_samples
        ),
        "account_age_days": rng.integers(
            1, 2000, n_samples
        ),
        "is_fraud": fraud
    })


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    baseline = generate_transactions(
        10000, seed=42
    )

    shifted = generate_transactions(
        5000, seed=43, drift=True
    )

    baseline.to_csv(
        "data/baseline.csv", index=False
    )

    shifted.to_csv(
        "data/shifted.csv", index=False
    )

    print("Datasets generated successfully!")
    print("Baseline records:", len(baseline))
    print("Shifted records:", len(shifted))

    print("\nBaseline sample:")
    print(baseline.head())