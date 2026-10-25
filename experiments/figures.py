"""Figures (PDF, vector) and error-analysis tables for all datasets with finished results."""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fmt_tau(t):
    """Threshold label: 3 decimals, or 2 significant digits below 0.001 (extended grid)."""
    t = float(t)
    return f"{t:.3f}" if t >= 0.001 else f"{t:.2g}"
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from common import BALANCING, CLASSIFIERS, RES, load

FIG = RES / "figures"
FIG.mkdir(exist_ok=True)
T = RES / "tables"
DS = [("ulb_time", "ULB (European cardholders)"), ("ieee_time", "IEEE-CIS (Vesta)"), ("sparkov_card", "Sparkov (simulated)")]
DS = [(n, t) for n, t in DS if (RES / n / "final_preds.npz").exists()]
NP = len(DS)
PREFIX = {"ulb": "ulb_time", "ieee": "ieee_time", "sparkov": "sparkov_card"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"LR": "#7f7f7f", "RF": "#2ca02c", "XGB": "#1f77b4", "LGBM": "#9467bd", "MLP": "#ff7f0e", "Stacking": "#d62728"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def best_per_clf(name):
    main = pd.read_csv(T / f"{name}_main.csv")
    return main.sort_values("cv_pr_auc_mean", ascending=False).groupby("classifier").head(1)


# 1. PR curves --------------------------------------------------------------
fig, axes = plt.subplots(1, NP, figsize=(3.5 * NP, 3.0))
for ax, (name, title) in zip(axes, DS):
    yte = load(name)["yte"].astype(int)
    for _, row in best_per_clf(name).iterrows():
        r = pickle.load(open(RES / name / f"{row.classifier}__{row.balancing}.pkl", "rb"))
        pr, rc, _ = precision_recall_curve(yte, r["test_pred"])
        ax.plot(rc, pr, color=COL[row.classifier], lw=1, label=f"{row.classifier}+{row.balancing} ({row.test_pr_auc:.3f})")
    f = np.load(RES / name / "final_preds.npz")
    pr, rc, _ = precision_recall_curve(yte, f["stack"])
    from sklearn.metrics import average_precision_score
    ax.plot(rc, pr, color=COL["Stacking"], lw=1.6, label=f"Stacking ({average_precision_score(yte, f['stack']):.3f})")
    ax.axhline(yte.mean(), color="k", ls=":", lw=0.8, label=f"No-skill ({yte.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.3, lw=0.4)
save(fig, "fig_pr_curves")

# 2. heatmap of test PR-AUC -------------------------------------------------
fig, axes = plt.subplots(1, NP, figsize=(3.5 * NP, 2.6))
for ax, (name, title) in zip(axes, DS):
    main = pd.read_csv(T / f"{name}_main.csv")
    M = main.pivot(index="classifier", columns="balancing", values="test_pr_auc").reindex(index=CLASSIFIERS, columns=BALANCING)
    im = ax.imshow(M.values, cmap="Blues", vmin=np.nanmin(M.values), vmax=np.nanmax(M.values))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmean(M.values) + 0.5 * np.nanstd(M.values) else "black")
    ax.set_xticks(range(len(BALANCING)), BALANCING, rotation=20, ha="right")
    ax.set_yticks(range(len(CLASSIFIERS)), CLASSIFIERS)
    ax.set_title(f"{title}: test PR-AUC")
    ax.set_xlabel("Balancing method")
    fig.colorbar(im, ax=ax, fraction=0.04, label="PR-AUC")
save(fig, "fig_heatmap_pr_auc")

# 3. threshold curves for the stacking model (log-scaled threshold axis) ----
from matplotlib.lines import Line2D
fig, axes = plt.subplots(1, NP, figsize=(3.5 * NP, 2.8))
grid = np.logspace(-5, 0, 400)
for ax, (name, title) in zip(axes, DS):
    f = np.load(RES / name / "final_preds.npz")
    y, p = f["yte"].astype(int), f["stack"]
    yb = y == 1
    order = np.argsort(p)
    ps, ys = p[order], yb[order]
    n_pos, n_neg = int(yb.sum()), int((~yb).sum())
    # positives/negatives with score >= t, via the sorted scores
    pos_below = np.r_[0, np.cumsum(ys)]
    neg_below = np.r_[0, np.cumsum(~ys)]
    k = np.searchsorted(ps, grid, side="left")
    tp = n_pos - pos_below[k]; fp = n_neg - neg_below[k]; fn = n_pos - tp
    prec = np.divide(tp, tp + fp, out=np.ones(len(grid)), where=(tp + fp) > 0)
    rec = tp / n_pos
    fpr = fp / n_neg
    f2 = np.divide(5 * tp, 5 * tp + 4 * fn + fp, out=np.zeros(len(grid)), where=tp > 0)
    ax.plot(grid, prec, color="#1f77b4", lw=1.2)
    ax.plot(grid, rec, color="#2ca02c", lw=1.2)
    ax.plot(grid, f2, color="#d62728", lw=1.2)
    ax.plot(grid, fpr, color="#7f7f7f", lw=1.2, ls="--")
    ax.axvline(float(f["tau_st"]), color="k", lw=0.8, ls="-.")
    ax.axvline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set(xlim=(1e-5, 1), ylim=(-0.02, 1.02), xlabel="Decision threshold $\\tau$ (log scale)", ylabel="Metric value",
           title=f"{title}\nstacking, $\\tau^*$ = {fmt_tau(f['tau_st'])}")
    ax.grid(alpha=0.3, lw=0.4, which="major")
handles = [Line2D([], [], color="#1f77b4", lw=1.2), Line2D([], [], color="#2ca02c", lw=1.2),
           Line2D([], [], color="#d62728", lw=1.2), Line2D([], [], color="#7f7f7f", lw=1.2, ls="--"),
           Line2D([], [], color="k", lw=0.8, ls="-."), Line2D([], [], color="k", lw=0.8, ls=":")]
labels = ["Precision", "Recall", "$F_2$", "FPR", "$\\tau^*$ (chosen on out-of-fold predictions)", "$\\tau$ = 0.5"]
fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
save(fig, "fig_threshold_curves")

# 4. confusion matrices + error analysis tables -------------------------------
fig, axes = plt.subplots(1, NP, figsize=(3.0 * NP, 2.4), gridspec_kw=dict(wspace=0.55))
err_rows = []
for ax, (name, title) in zip(axes, DS):
    f = np.load(RES / name / "final_preds.npz")
    d = load(name)
    y, p, tau = f["yte"].astype(int), f["stack"], float(f["tau_st"])
    yh = (p >= tau).astype(int)
    cm = confusion_matrix(y, yh, labels=[0, 1])
    ax.imshow(cm, cmap="Greys", alpha=0.0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9)
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"{title}\n(stacking, $\\tau^*$ = {fmt_tau(tau)})")
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
    ax.axhline(0.5, color="k", lw=0.5); ax.axvline(0.5, color="k", lw=0.5)
    amt = d["amt_te"]
    fr = y == 1
    err_rows.append(dict(dataset=name, group="all frauds", n_fraud=int(fr.sum()), detected=int((yh[fr] == 1).sum()),
                         recall=float(yh[fr].mean()), fraud_amount=float(amt[fr].sum()),
                         missed_amount_share=float(amt[fr & (yh == 0)].sum() / amt[fr].sum()),
                         false_positives=int(((yh == 1) & ~fr).sum())))
    bins = [0, 10, 100, 500, np.inf]
    labels = ["< 10", "10-100", "100-500", "> 500"]
    cat = pd.cut(amt, bins=bins, labels=labels, right=False)
    for lab in labels:
        m = fr & (cat == lab)
        if m.sum():
            err_rows.append(dict(dataset=name, group=f"amount {lab}", n_fraud=int(m.sum()), detected=int(yh[m].sum()),
                                 recall=float(yh[m].mean()), fraud_amount=float(amt[m].sum()),
                                 false_positives=int(((yh == 1) & ~fr & (cat == lab)).sum())))
    if "cat_te" in d:
        for pc in sorted(set(d["cat_te"])):
            m = fr & (d["cat_te"] == pc)
            if m.sum():
                err_rows.append(dict(dataset=name, group=f"category {pc}", n_fraud=int(m.sum()), detected=int(yh[m].sum()),
                                     recall=float(yh[m].mean()), n_transactions=int((d["cat_te"] == pc).sum()),
                                     false_positives=int(((yh == 1) & ~fr & (d["cat_te"] == pc)).sum())))
    if "prod_te" in d:
        for pc in sorted(set(d["prod_te"])):
            m = fr & (d["prod_te"] == pc)
            if m.sum():
                err_rows.append(dict(dataset=name, group=f"ProductCD {pc}", n_fraud=int(m.sum()), detected=int(yh[m].sum()),
                                     recall=float(yh[m].mean()), n_transactions=int((d["prod_te"] == pc).sum()),
                                     false_positives=int(((yh == 1) & ~fr & (d["prod_te"] == pc)).sum())))
save(fig, "fig_confusion")
pd.DataFrame(err_rows).to_csv(T / "error_analysis.csv", index=False)

# 5. protocol comparison (time / random / leaky) -------------------------------
rows = []
for ds in ("ulb", "ieee", "sparkov"):
    f = T / f"{ds}_sensitivity.csv"
    if f.exists():
        rows.append(pd.read_csv(f))
if rows:
    s = pd.concat(rows)
    s = s[s.config.str.contains("LR") == False]  # noqa: E712  best configuration only
    dss = [ds for ds in ("ulb", "ieee", "sparkov") if ds in set(s.dataset)]
    npr = [int((s.dataset == ds).sum()) for ds in dss]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.5 * len(dss), 2.6), sharey=True,
                             gridspec_kw=dict(width_ratios=npr))
    for ax, ds in zip(np.atleast_1d(axes), dss):
        q = s[s.dataset == ds]
        x = np.arange(len(q))
        w = 0.25
        for i, (m, lab) in enumerate([("pr_auc", "PR-AUC"), ("f2", "$F_2$ at 0.5"), ("mcc", "MCC at 0.5")]):
            ax.bar(x + (i - 1) * w, q[m].values, w, label=lab)
            for xi, v in zip(x, q[m].values):
                ax.text(xi + (i - 1) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=5.5)
        ax.set_xticks(x, [p.replace(" (main)", "").replace("SMOTE before split (leaky)", "SMOTE before\nsplit (leaky)").replace("stratified ", "")
                          .replace(" (last 6 months)", "\n(last 6 months)") for p in q.protocol], fontsize=6.5)
        ax.set(title=dict(DS).get(PREFIX[ds], ds), ylim=(0, 1.1), ylabel="Metric value")
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17))
    save(fig, "fig_protocol_comparison")
print("figures done")
