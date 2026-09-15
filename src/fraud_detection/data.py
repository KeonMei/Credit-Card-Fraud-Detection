"""Loading of the two Kaggle datasets and time-aware train/validation/test split.

Supported datasets
------------------
ulb   - "Credit Card Fraud Detection" (Machine Learning Group, ULB).
        File: creditcard.csv. 284,807 transactions, 492 frauds (0.17%).
        Columns: Time, V1..V28 (PCA components), Amount, Class.
ieee  - "IEEE-CIS Fraud Detection" (Vesta Corporation).
        Files: train_transaction.csv + train_identity.csv (joined on TransactionID).
        590,540 transactions, ~3.5% fraud. Kaggle's test set has no labels,
        so all evaluation here is done on a time-based hold-out of the train files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target: str
    time_col: str
    amount_col: str
    id_col: str | None


SPECS = {
    "ulb": DatasetSpec("ulb", target="Class", time_col="Time", amount_col="Amount", id_col=None),
    "ieee": DatasetSpec(
        "ieee", target="isFraud", time_col="TransactionDT", amount_col="TransactionAmt", id_col="TransactionID"
    ),
}


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 roughly halves memory for the 400+ column IEEE table."""
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def load_ulb(data_dir: str | Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "creditcard.csv", nrows=nrows)


def read_ieee_files(
    transaction_path: str | Path, identity_path: str | Path | None = None, nrows: int | None = None
) -> pd.DataFrame:
    transactions = _downcast(pd.read_csv(transaction_path, nrows=nrows))
    if identity_path is not None and Path(identity_path).exists():
        identity = _downcast(pd.read_csv(identity_path))
        # Kaggle's test_identity uses "id-01" instead of "id_01"
        identity.columns = [c.replace("-", "_") for c in identity.columns]
        transactions = transactions.merge(identity, on="TransactionID", how="left")
    return transactions


def load_ieee(data_dir: str | Path, nrows: int | None = None, prefix: str = "train") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return read_ieee_files(data_dir / f"{prefix}_transaction.csv", data_dir / f"{prefix}_identity.csv", nrows)


def load_dataset(name: str, data_dir: str | Path, nrows: int | None = None) -> pd.DataFrame:
    if name == "ulb":
        return load_ulb(data_dir, nrows)
    if name == "ieee":
        return load_ieee(data_dir, nrows)
    raise ValueError(f"Unknown dataset {name!r}, expected one of {list(SPECS)}")


def split_data(
    df: pd.DataFrame,
    spec: DatasetSpec,
    strategy: str = "time",
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / validation / test.

    "time"   - chronological split: the model is trained on the past and evaluated
               on the future, as it would be in production. This is the default and
               the recommended setting for the thesis.
    "random" - stratified random split. Common in papers, but optimistic: it leaks
               future information and (for IEEE) the same card into train and test.
               Keep it only to compare against published numbers.
    """
    if strategy == "time":
        df = df.sort_values(spec.time_col, kind="stable").reset_index(drop=True)
        n = len(df)
        train_end = int(n * (1 - val_size - test_size))
        val_end = int(n * (1 - test_size))
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    if strategy == "random":
        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df[spec.target], random_state=random_state
        )
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            stratify=train_val[spec.target],
            random_state=random_state,
        )
        return train, val, test

    raise ValueError(f"Unknown split strategy {strategy!r}")
