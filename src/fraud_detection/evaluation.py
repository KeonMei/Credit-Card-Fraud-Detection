"""Metrics, threshold selection and plots for highly imbalanced fraud data.

Why not accuracy: with 0.17% fraud, predicting "legitimate" for everything gives
99.83% accuracy. ROC-AUC is also over-optimistic under heavy imbalance, so the
primary metric here is PR-AUC (average precision) — see Saito & Rehmsmeier (2015).
Operational metrics follow Le Borgne et al., "Reproducible Machine Learning for
Credit Card Fraud Detection — Practical Handbook" (2022):
precision@k models a fraud team that can only review k alerts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_best_threshold(y_true, proba, beta: float = 1.0) -> float:
    """Threshold maximising F-beta on the given (validation) data.

    beta > 1 favours recall (missed fraud is expensive), beta < 1 favours precision
    (every alert costs analyst time / annoys the customer).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    denom = beta**2 * precision + recall
    fbeta = np.divide((1 + beta**2) * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(fbeta)])


def precision_at_k(y_true, proba, k: int) -> float:
    """Share of frauds among the k most suspicious transactions."""
    k = min(k, len(proba))
    top_k = np.argsort(-np.asarray(proba), kind="stable")[:k]
    return float(np.asarray(y_true)[top_k].mean())


def compute_metrics(y_true, proba, threshold: float, amount=None, k: int = 100) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        f"precision@{k}": precision_at_k(y_true, proba, k),
        "alert_rate": float(y_pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    if amount is not None:
        # Business view: what share of the stolen money would be stopped
        amount = np.asarray(amount, dtype=float)
        fraud_amount = amount[y_true == 1].sum()
        caught_amount = amount[(y_true == 1) & (y_pred == 1)].sum()
        metrics["fraud_amount_recall"] = float(caught_amount / fraud_amount) if fraud_amount > 0 else float("nan")

    return {k_: float(v) if isinstance(v, (np.floating, float)) else v for k_, v in metrics.items()}


# ---------------------------------------------------------------------- plots


def plot_curves(y_true, probas: dict[str, np.ndarray], out_path: Path, title: str = "") -> None:
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))
    baseline = float(np.mean(y_true))

    for name, proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax_pr.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_true, proba):.3f})")
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")

    ax_pr.axhline(baseline, color="grey", linestyle="--", linewidth=1, label=f"random ({baseline:.4f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve", xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    for ax in (ax_pr, ax_roc):
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(metrics: dict, out_path: Path, title: str = "") -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="white" if value > cm.max() / 2 else "black")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["legit", "fraud"],
        yticklabels=["legit", "fraud"],
        xlabel="Predicted",
        ylabel="Actual",
        title=title or f"Confusion matrix (threshold={metrics['threshold']:.3f})",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, out_path: Path, top_n: int = 25, title: str = "") -> None:
    top = importance.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(top))))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set(xlabel="mean |SHAP value|" if title.endswith("SHAP") else "importance", title=title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
