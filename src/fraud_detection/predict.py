"""Score new transactions with a saved model.

Examples
--------
python -m fraud_detection.predict --model artifacts/ulb/model.joblib --input new_transactions.csv
python -m fraud_detection.predict --model artifacts/ieee/model.joblib \
    --input data/ieee/test_transaction.csv --identity data/ieee/test_identity.csv --output scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import SPECS, read_ieee_files


class FraudScorer:
    """Thin wrapper around a saved bundle: raw transactions in, fraud scores out."""

    def __init__(self, bundle_path: str | Path):
        bundle = joblib.load(bundle_path)
        self.dataset = bundle["dataset"]
        self.builder = bundle["builder"]
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.spec = SPECS[self.dataset]

    def score(self, transactions: pd.DataFrame) -> pd.DataFrame:
        X = self.builder.transform(transactions)
        proba = self.model.predict_proba(X)
        result = pd.DataFrame(index=transactions.index)
        if self.spec.id_col and self.spec.id_col in transactions:
            result[self.spec.id_col] = transactions[self.spec.id_col]
        result["fraud_probability"] = proba
        result["is_fraud_alert"] = (proba >= self.threshold).astype(np.int8)
        return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="CSV with raw transactions (same columns as training)")
    p.add_argument("--identity", type=Path, default=None, help="IEEE only: identity CSV to join on TransactionID")
    p.add_argument("--output", type=Path, default=Path("scored.csv"))
    args = p.parse_args(argv)

    scorer = FraudScorer(args.model)
    df = read_ieee_files(args.input, args.identity) if scorer.dataset == "ieee" else pd.read_csv(args.input)

    scored = scorer.score(df)
    scored.to_csv(args.output, index=False)
    n_alerts = int(scored["is_fraud_alert"].sum())
    print(f"Scored {len(scored)} transactions: {n_alerts} alerts ({100 * n_alerts / len(scored):.2f}%) -> {args.output}")


if __name__ == "__main__":
    main()
