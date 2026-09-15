"""Train, compare and save fraud detection models.

Examples
--------
python -m fraud_detection.train --dataset ulb  --data-dir data/ulb
python -m fraud_detection.train --dataset ieee --data-dir data/ieee --models logreg xgb lgbm
python -m fraud_detection.train --dataset ulb  --data-dir data/ulb --imbalance smote --split random

Protocol
--------
1. chronological split: train (60%) -> validation (20%) -> test (20%)
2. feature builder is fitted on train only
3. each model is trained on train; boosting models early-stop on validation
4. the decision threshold is chosen on validation (max F-beta)
5. final numbers are reported on test, which is used exactly once
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .data import SPECS, load_dataset, split_data
from .evaluation import (
    compute_metrics,
    find_best_threshold,
    plot_confusion_matrix,
    plot_curves,
    plot_feature_importance,
)
from .features import make_feature_builder
from .models import BOOSTING_MODELS, IMBALANCE_STRATEGIES, MODEL_NAMES, FraudModel

log = logging.getLogger("fraud_detection")

DEFAULT_MODELS = {"ulb": ["logreg", "rf", "xgb", "lgbm"], "ieee": ["logreg", "xgb", "lgbm"]}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(SPECS), required=True)
    p.add_argument("--data-dir", type=Path, required=True, help="folder with the Kaggle CSV files")
    p.add_argument("--output-dir", type=Path, default=None, help="default: artifacts/<dataset>")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    p.add_argument("--imbalance", choices=IMBALANCE_STRATEGIES, default="weight")
    p.add_argument("--split", choices=["time", "random"], default="time")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=1.0, help="F-beta used to pick the threshold (2 = favour recall)")
    p.add_argument("--top-k", type=int, default=100, help="k for precision@k")
    p.add_argument("--nrows", type=int, default=None, help="read only the first N rows (quick experiments)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> pd.DataFrame:
    spec = SPECS[args.dataset]
    models = args.models or DEFAULT_MODELS[args.dataset]
    out_dir = args.output_dir or Path("artifacts") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s from %s", args.dataset, args.data_dir)
    df = load_dataset(args.dataset, args.data_dir, nrows=args.nrows)
    log.info("%d transactions, %d frauds (%.3f%%)", len(df), df[spec.target].sum(), 100 * df[spec.target].mean())

    train_df, val_df, test_df = split_data(df, spec, args.split, args.val_size, args.test_size, args.seed)
    del df
    for part, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        log.info("  %-5s %8d rows, %6d frauds (%.3f%%)", name, len(part), part[spec.target].sum(), 100 * part[spec.target].mean())

    t0 = time.perf_counter()
    builder = make_feature_builder(args.dataset).fit(train_df)
    X_train, X_val, X_test = (builder.transform(d) for d in (train_df, val_df, test_df))
    y_train, y_val, y_test = (d[spec.target].to_numpy() for d in (train_df, val_df, test_df))
    amount_val, amount_test = val_df[spec.amount_col].to_numpy(), test_df[spec.amount_col].to_numpy()
    log.info("Built %d features in %.1fs", X_train.shape[1], time.perf_counter() - t0)

    rows, test_probas, fitted = [], {}, {}
    for name in models:
        log.info("Training %s (imbalance=%s)", name, args.imbalance)
        t0 = time.perf_counter()
        model = FraudModel(name, imbalance=args.imbalance, random_state=args.seed).fit(X_train, y_train, X_val, y_val)
        train_time = time.perf_counter() - t0

        p_val = model.predict_proba(X_val)
        threshold = find_best_threshold(y_val, p_val, beta=args.beta)
        p_test = model.predict_proba(X_test)

        for split_name, y, p, amount in [("val", y_val, p_val, amount_val), ("test", y_test, p_test, amount_test)]:
            m = compute_metrics(y, p, threshold, amount=amount, k=args.top_k)
            rows.append({"model": name, "split": split_name, "train_time_s": round(train_time, 1),
                         "best_iteration": model.best_iteration, **m})

        test_probas[name] = p_test
        fitted[name] = (model, threshold)
        test_row = rows[-1]
        log.info("  %s: test PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f (%.1fs)",
                 name, test_row["pr_auc"], test_row["roc_auc"], test_row["f1"], train_time)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "metrics.csv", index=False)

    # Model selection uses validation only — choosing by test score would leak.
    val_results = results[results["split"] == "val"].set_index("model")
    best_name = val_results["pr_auc"].idxmax()
    best_model, best_threshold = fitted[best_name]
    best_test = results[(results["model"] == best_name) & (results["split"] == "test")].iloc[0].to_dict()
    log.info("Best model on validation: %s", best_name)

    title = f"{args.dataset.upper()} — test set ({args.split} split, imbalance={args.imbalance})"
    plot_curves(y_test, test_probas, out_dir / "curves_test.png", title=title)
    plot_confusion_matrix(best_test, out_dir / f"confusion_matrix_{best_name}.png",
                          title=f"{best_name} — test (threshold={best_threshold:.3f})")
    importance = best_model.feature_importance(X_val)
    importance.to_csv(out_dir / f"feature_importance_{best_name}.csv", header=["importance"])
    method = "SHAP" if best_name in BOOSTING_MODELS else "model"
    plot_feature_importance(importance, out_dir / f"feature_importance_{best_name}.png",
                            title=f"{best_name} feature importance — {method}")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["models"] = models
    bundle = {
        "dataset": args.dataset,
        "model_name": best_name,
        "builder": builder,
        "model": best_model,
        "threshold": best_threshold,
        "test_metrics": best_test,
        "config": config,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, out_dir / "model.joblib")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k not in ("builder", "model")}, f, indent=2, default=str)

    cols = ["model", "pr_auc", "roc_auc", "precision", "recall", "f1", f"precision@{args.top_k}",
            "fraud_amount_recall", "threshold", "train_time_s"]
    print("\nTest set results (threshold tuned on validation):")
    print(results[results["split"] == "test"][cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model (by validation PR-AUC): {best_name}. Artifacts saved to {out_dir.resolve()}")
    return results


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
