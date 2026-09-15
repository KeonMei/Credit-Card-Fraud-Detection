"""Feature engineering.

Every builder follows the sklearn-style fit/transform contract: all statistics
(frequencies, group means, category codes) are learned on the training period only
and then applied to validation/test. Computing them on the full data — as many
Kaggle notebooks do — leaks information from the future into training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SPECS

SECONDS_PER_DAY = 86_400


class ULBFeatureBuilder:
    """ULB data is already PCA-transformed (V1..V28), so there is little to engineer.

    `Time` is seconds since the first transaction of a 2-day window: as a raw value it
    only encodes "how far into the dataset", which doesn't generalise, so it is
    replaced by hour of day. `Amount` is heavy-tailed, so a log version is added.
    """

    spec = SPECS["ulb"]

    def fit(self, df: pd.DataFrame) -> ULBFeatureBuilder:
        self.feature_names_ = list(self._build(df.head(1)).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._build(df)[self.feature_names_]

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns=[self.spec.target, self.spec.time_col], errors="ignore").copy()
        X["hour"] = (df[self.spec.time_col] % SECONDS_PER_DAY) / 3600
        X["amount_log"] = np.log1p(df[self.spec.amount_col])
        return X


class IEEEFeatureBuilder:
    """Features for IEEE-CIS, based on the ideas of the top Kaggle solutions.

    The key idea: the dataset has no explicit card/client id, but one can be
    approximated. D1 is "days since the card was first used", so
    `transaction_day - D1` is constant for a given card; together with card1
    (card number fragment) and addr1 (billing region) it gives a pseudo client id
    `uid`. Fraud labels in this dataset are assigned per client, so behavioural
    aggregates over `uid` are very informative.
    """

    spec = SPECS["ieee"]

    FREQ_COLS = ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain", "card1_addr1", "uid"]
    GROUP_COLS = ["card1", "card1_addr1", "uid"]
    # Engineered keys used for encoding only; raw values are not model features
    HELPER_COLS = ["card1_addr1", "uid", "D1n", "day"]

    def fit(self, df: pd.DataFrame) -> IEEEFeatureBuilder:
        base = self._stateless(df)
        n = len(base)

        self.freq_maps_ = {
            col: base[col].astype(str).value_counts() / n for col in self.FREQ_COLS if col in base
        }
        self.amount_stats_ = {
            col: base.groupby(base[col].astype(str))[self.spec.amount_col].agg(["mean", "std"])
            for col in self.GROUP_COLS
            if col in base
        }
        self.categorical_cols_ = [
            c for c in base.columns if c not in self.HELPER_COLS and not pd.api.types.is_numeric_dtype(base[c])
        ]
        self.category_codes_ = {
            col: {value: code for code, value in enumerate(base[col].dropna().unique())}
            for col in self.categorical_cols_
        }

        self.feature_names_ = list(self._stateful(base.head(1)).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._stateful(self._stateless(df))[self.feature_names_]

    def _stateless(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.spec
        X = df.drop(columns=[s.target, s.id_col], errors="ignore").copy()
        X["n_missing"] = X.isna().sum(axis=1).astype(np.float32)

        dt = X.pop(s.time_col)
        X["day"] = np.floor(dt / SECONDS_PER_DAY)
        X["hour"] = np.floor(dt / 3600) % 24
        X["weekday"] = X["day"] % 7

        amount = X[s.amount_col]
        X["amount_log"] = np.log1p(amount)
        # Non-zero cents often mean the amount was converted from a foreign currency
        X["amount_cents"] = (amount - np.floor(amount)).round(3)

        if "card1" in X and "addr1" in X:
            X["card1_addr1"] = X["card1"].astype(str) + "_" + X["addr1"].astype(str)
            if "D1" in X:
                X["D1n"] = X["day"] - X["D1"]
                X["uid"] = X["card1_addr1"] + "_" + X["D1n"].astype(str)
        return X

    def _stateful(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        amount = X[self.spec.amount_col]

        for col, freq in self.freq_maps_.items():
            X[f"{col}_freq"] = X[col].astype(str).map(freq).fillna(0).astype(np.float32)

        for col, stats in self.amount_stats_.items():
            key = X[col].astype(str)
            mean = key.map(stats["mean"]).astype(np.float32)
            X[f"amount_mean_by_{col}"] = mean
            X[f"amount_std_by_{col}"] = key.map(stats["std"]).astype(np.float32)
            X[f"amount_to_mean_{col}"] = (amount / mean).astype(np.float32)

        for col, codes in self.category_codes_.items():
            X[col] = X[col].map(codes).fillna(-1).astype(np.int32)

        return X.drop(columns=[c for c in self.HELPER_COLS if c in X])


def make_feature_builder(dataset: str):
    builders = {"ulb": ULBFeatureBuilder, "ieee": IEEEFeatureBuilder}
    if dataset not in builders:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return builders[dataset]()
