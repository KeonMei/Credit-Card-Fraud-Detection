"""Small synthetic stand-ins for the Kaggle files, so tests run without downloading data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ulb_like(n: int = 6000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    df = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172_800, n))})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n)
    df.loc[y == 1, ["V14", "V17"]] -= 3  # frauds are separable on a few components
    df["Amount"] = rng.lognormal(3, 1.5, n).round(2)
    df["Class"] = y
    return df


def make_ieee_like(n: int = 6000, fraud_rate: float = 0.05, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_cards = 800
    card = rng.integers(0, n_cards, n)
    card_is_fraud = rng.random(n_cards) < fraud_rate  # labels are per client, like in IEEE-CIS
    y = card_is_fraud[card].astype(int)
    dt = np.sort(rng.integers(86_400, 86_400 * 180, n))
    day = dt // 86_400
    first_day = rng.integers(0, 60, n_cards)

    tx = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + n),
        "isFraud": y,
        "TransactionDT": dt,
        "TransactionAmt": rng.lognormal(4, 1, n).round(3) * (1 + y),
        "ProductCD": rng.choice(["W", "C", "H", "R", "S"], n),
        "card1": 1000 + card,
        "card2": np.where(rng.random(n) < 0.1, np.nan, rng.integers(100, 600, n)),
        "card4": rng.choice(["visa", "mastercard", None], n),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": np.where(rng.random(n) < 0.1, np.nan, 100 + card % 50),
        "P_emaildomain": np.where(y == 1, "anonymous.com", rng.choice(["gmail.com", "yahoo.com"], n)),
        "D1": (day - first_day[card]).clip(0).astype(float),
        "C1": rng.poisson(2, n).astype(float),
        "V258": rng.normal(size=n) + y,
        "M4": rng.choice(["M0", "M1", "M2", None], n),
    })
    has_identity = rng.random(n) < 0.3
    identity = pd.DataFrame({
        "TransactionID": tx.loc[has_identity, "TransactionID"].to_numpy(),
        "id_01": rng.normal(size=has_identity.sum()),
        "id_31": rng.choice(["chrome 63.0", "safari generic", "edge 16.0"], has_identity.sum()),
        "DeviceType": rng.choice(["desktop", "mobile"], has_identity.sum()),
    })
    return tx, identity


@pytest.fixture
def ulb_dir(tmp_path):
    make_ulb_like().to_csv(tmp_path / "creditcard.csv", index=False)
    return tmp_path


@pytest.fixture
def ieee_dir(tmp_path):
    tx, identity = make_ieee_like()
    tx.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)
    return tmp_path
