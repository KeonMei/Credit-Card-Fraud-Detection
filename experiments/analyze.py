"""Results for one dataset: main table, statistical tests, RQ1 comparisons, stacking, threshold analysis.
Usage: python analyze.py <dataset>   (e.g. ulb_time)
"""
import json
import pickle
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from scipy.stats import spearmanr

from common import BALANCING, CLASSIFIERS, RES, all_metrics, best_f2_threshold, get_folds, load
from stats_utils import bootstrap_ci, delong_test, holm, paired_bootstrap

warnings.filterwarnings("ignore")
name = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rdir = RES / name
tdir = RES / "tables"
tdir.mkdir(exist_ok=True)
d = load(name)
y_tr, yte = d["ytr"].astype(int), d["yte"].astype(int)
folds = get_folds(d)
grouped = "fold_tr" in d  # Sparkov: card-grouped folds instead of time blocks
oof_mask = np.zeros(len(y_tr), bool)
for _, va in folds:
    oof_mask[va] = True

R = {}
for c, b in product(CLASSIFIERS, BALANCING):
    f = rdir / f"{c}__{b}.pkl"
    if f.exists():
        R[(c, b)] = pickle.load(open(f, "rb"))

# ------------------------------------------------------------- main table
for r in R.values():  # F2-optimal threshold on out-of-fold predictions (never on the test set)
    r["tau"] = best_f2_threshold(y_tr[oof_mask], r["oof"][oof_mask])[0]
main_f = tdir / f"{name}_main.csv"
rows = []
for (c, b), r in ([] if main_f.exists() else R.items()):
    tau = r["tau"]
    m05 = all_metrics(yte, r["test_pred"], 0.5)
    mt = all_metrics(yte, r["test_pred"], tau)
    ci = bootstrap_ci(yte, r["test_pred"], tau, metrics=("pr_auc",), B=B, seed=1)
    rows.append(dict(classifier=c, balancing=b, ratio=r["ratio"], cv_pr_auc_mean=r["cv_mean"], cv_pr_auc_std=r["cv_std"],
                     test_pr_auc=mt["pr_auc"], pr_auc_lo=ci["pr_auc"][0], pr_auc_hi=ci["pr_auc"][1],
                     test_roc_auc=mt["roc_auc"], tau=tau,
                     **{f"{k}_tau": v for k, v in mt.items() if k in ("precision", "recall", "f1", "f2", "mcc", "fpr", "tp", "fp", "fn", "tn")},
                     **{f"{k}_05": v for k, v in m05.items() if k in ("precision", "recall", "f1", "f2", "mcc", "fpr")},
                     fit_seconds=r["fit_seconds_full"], search_seconds=r["search_seconds"], params=json.dumps(r["params"])))
if main_f.exists():  # bootstrap-heavy sections are reused when the analysis is restarted
    main = pd.read_csv(main_f)
else:
    main = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)
    main.to_csv(main_f, index=False)
print(main[["classifier", "balancing", "cv_pr_auc_mean", "cv_pr_auc_std", "test_pr_auc", "test_roc_auc", "f2_tau", "mcc_tau",
            "fpr_tau"]].round(4).to_string(), flush=True)

# ------------------------------------------------------------- best vs all (bootstrap + DeLong + Holm)
best_key = (main.loc[0, "classifier"], main.loc[0, "balancing"])
rb = R[best_key]
cmp_rows = []
skip_cmp = (tdir / f"{name}_best_vs_all.csv").exists()
for key, r in ([] if skip_cmp else R.items()):
    if key == best_key:
        continue
    pb = paired_bootstrap(yte, rb["test_pred"], rb["tau"], r["test_pred"], r["tau"], B=B, seed=2)
    a1, a2, z, p = delong_test(yte, rb["test_pred"], r["test_pred"])
    cmp_rows.append(dict(other_classifier=key[0], other_balancing=key[1],
                         d_pr_auc=pb["pr_auc"]["diff"], d_pr_auc_lo=pb["pr_auc"]["lo"], d_pr_auc_hi=pb["pr_auc"]["hi"],
                         p_pr_auc=pb["pr_auc"]["p"], d_f2=pb["f2"]["diff"], p_f2=pb["f2"]["p"],
                         d_mcc=pb["mcc"]["diff"], p_mcc=pb["mcc"]["p"], d_roc_auc=a1 - a2, p_delong=p))
if not skip_cmp:
    cmp = pd.DataFrame(cmp_rows)
    for col in ("p_pr_auc", "p_f2", "p_mcc", "p_delong"):
        cmp[col + "_holm"] = holm(cmp[col].values)
    cmp.insert(0, "best", f"{best_key[0]}+{best_key[1]}")
    cmp.to_csv(tdir / f"{name}_best_vs_all.csv", index=False)
print("best:", best_key, flush=True)

# ------------------------------------------------------------- RQ1: each balancing vs none, per classifier
rq1 = []
skip_rq1 = (tdir / f"{name}_rq1_balancing_vs_none.csv").exists()
for c in ([] if skip_rq1 else CLASSIFIERS):
    if (c, "none") not in R:
        continue
    r0 = R[(c, "none")]
    for b in BALANCING[1:]:
        if (c, b) not in R:
            continue
        r = R[(c, b)]
        pb = paired_bootstrap(yte, r["test_pred"], r["tau"], r0["test_pred"], r0["tau"], B=B, seed=3)
        a1, a2, z, p = delong_test(yte, r["test_pred"], r0["test_pred"])
        rq1.append(dict(classifier=c, balancing=b, d_pr_auc=pb["pr_auc"]["diff"], lo=pb["pr_auc"]["lo"], hi=pb["pr_auc"]["hi"],
                        p_pr_auc=pb["pr_auc"]["p"], d_f2=pb["f2"]["diff"], p_f2=pb["f2"]["p"], d_roc_auc=a1 - a2, p_delong=p))
if not skip_rq1:
    rq1 = pd.DataFrame(rq1)
    for col in ("p_pr_auc", "p_f2", "p_delong"):
        rq1[col + "_holm"] = holm(rq1[col].values)
    rq1.to_csv(tdir / f"{name}_rq1_balancing_vs_none.csv", index=False)

# ------------------------------------------------------------- RQ2: agreement between metrics
metric_cols = ["test_pr_auc", "test_roc_auc", "f1_tau", "f2_tau", "mcc_tau", "precision_tau", "recall_tau"]
rank_corr = pd.DataFrame([[spearmanr(main[a], main[b])[0] for b in metric_cols] for a in metric_cols],
                         index=metric_cols, columns=metric_cols)
rank_corr.to_csv(tdir / f"{name}_rq2_metric_rank_correlation.csv")

# ------------------------------------------------------------- stacking
best_bal = best_key[1]
base_keys = [(c, best_bal) for c in ("XGB", "LGBM", "RF", "MLP") if (c, best_bal) in R]
Z = np.column_stack([R[k]["oof"] for k in base_keys])
Zte = np.column_stack([R[k]["test_pred"] for k in base_keys])
blocks = [va for _, va in folds]  # blocks 2..6 of the training period
ALPHAS = [1, 5, 10, 25, 50, 100]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]


def meta_model(alpha, lam, n):
    # (1/N) sum weighted log-loss + lam*||w||^2  <=>  sklearn C = 1 / (2 * lam * N)
    return LogisticRegression(C=1.0 / (2 * lam * n), class_weight={0: 1, 1: alpha}, max_iter=2000)


grid = []
meta_val = list(range(len(blocks))) if grouped else list(range(1, len(blocks)))
meta_train = (lambda k: [j for j in range(len(blocks)) if j != k]) if grouped else (lambda k: list(range(k)))
val_idx = np.concatenate([blocks[k] for k in meta_val])
for a, lam in product(ALPHAS, LAMBDAS):
    scores, pv = [], []
    for k in meta_val:  # time data: train on earlier blocks only; grouped data: on the other folds
        tr = np.concatenate([blocks[j] for j in meta_train(k)])
        va = blocks[k]
        mm = meta_model(a, lam, len(tr)).fit(Z[tr], y_tr[tr])
        pv.append(mm.predict_proba(Z[va])[:, 1])
        scores.append(average_precision_score(y_tr[va], pv[-1]))
    t_g, f2_g = best_f2_threshold(y_tr[val_idx], np.concatenate(pv))
    grid.append(dict(alpha=a, lam=lam, meta_cv_pr_auc=float(np.mean(scores)), std=float(np.std(scores, ddof=1)),
                     oof_f2_at_best_grid_threshold=f2_g, best_grid_threshold=t_g))
grid = pd.DataFrame(grid).sort_values("meta_cv_pr_auc", ascending=False).reset_index(drop=True)
# PR-AUC depends only on the ranking, so strongly regularised settings can tie with weak ones while their
# probabilities collapse below the 0.001 threshold grid. Among settings within 0.001 of the best PR-AUC,
# the one with the highest out-of-fold F2 on the threshold grid is chosen.
near = grid[grid.meta_cv_pr_auc >= grid.meta_cv_pr_auc.max() - 0.001]
sel = near.sort_values(["oof_f2_at_best_grid_threshold", "meta_cv_pr_auc"], ascending=False).iloc[0]
grid["selected"] = (grid.alpha == sel.alpha) & (grid.lam == sel.lam)
grid.to_csv(tdir / f"{name}_stacking_grid.csv", index=False)
a_best, l_best = sel["alpha"], sel["lam"]
meta_oof = np.full(len(y_tr), np.nan)
for k in meta_val:
    tr = np.concatenate([blocks[j] for j in meta_train(k)])
    meta_oof[blocks[k]] = meta_model(a_best, l_best, len(tr)).fit(Z[tr], y_tr[tr]).predict_proba(Z[blocks[k]])[:, 1]
mo = ~np.isnan(meta_oof)
tau_st, _ = best_f2_threshold(y_tr[mo], meta_oof[mo])
all_tr = np.concatenate(blocks)
meta = meta_model(a_best, l_best, len(all_tr)).fit(Z[all_tr], y_tr[all_tr])
p_stack = meta.predict_proba(Zte)[:, 1]
avg_oof, p_avg = Z.mean(1), Zte.mean(1)
tau_avg, _ = best_f2_threshold(y_tr[mo], avg_oof[mo])
tau_best_single = rb["tau"]

stack_rows = []
for label, p, tau in [("stacking", p_stack, tau_st), ("plain average", p_avg, tau_avg),
                      (f"best single ({best_key[0]}+{best_key[1]})", rb["test_pred"], tau_best_single)]:
    for tname, t in (("0.5", 0.5), ("tau*", tau)):
        m = all_metrics(yte, p, t)
        ci = bootstrap_ci(yte, p, t, B=B, seed=4)
        stack_rows.append(dict(model=label, threshold_rule=tname, threshold=t, **{k: m[k] for k in
                          ("pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "mcc", "fpr", "tp", "fp", "fn", "tn")},
                          pr_auc_lo=ci["pr_auc"][0], pr_auc_hi=ci["pr_auc"][1], f2_lo=ci["f2"][0], f2_hi=ci["f2"][1],
                          mcc_lo=ci["mcc"][0], mcc_hi=ci["mcc"][1]))
stack = pd.DataFrame(stack_rows)
stack.to_csv(tdir / f"{name}_stacking_vs_single.csv", index=False)

tests = []
for label, p, tau in [("stacking vs best single", p_stack, tau_st), ("stacking vs plain average", p_avg, tau_avg)]:
    other_p, other_t = (rb["test_pred"], tau_best_single) if "single" in label else (p_avg, tau_avg)
    pa_ = p_stack
    pb = paired_bootstrap(yte, pa_, tau_st, other_p, other_t, B=B, seed=5)
    a1, a2, z, pv = delong_test(yte, pa_, other_p)
    tests.append(dict(comparison=label, d_pr_auc=pb["pr_auc"]["diff"], lo=pb["pr_auc"]["lo"], hi=pb["pr_auc"]["hi"],
                      p_pr_auc=pb["pr_auc"]["p"], d_f2=pb["f2"]["diff"], p_f2=pb["f2"]["p"],
                      d_mcc=pb["mcc"]["diff"], p_mcc=pb["mcc"]["p"], d_roc_auc=a1 - a2, p_delong=pv))
tests = pd.DataFrame(tests)
tests.to_csv(tdir / f"{name}_stacking_tests.csv", index=False)


def recall_at_fpr(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target, fpr, tpr))


thr_rows = []
for label, p in [("stacking", p_stack), (f"best single", rb["test_pred"])]:
    thr_rows.append(dict(model=label, recall_at_fpr_0_1pct=recall_at_fpr(yte, p, 0.001),
                         recall_at_fpr_1pct=recall_at_fpr(yte, p, 0.01)))
pd.DataFrame(thr_rows).to_csv(tdir / f"{name}_recall_at_fpr.csv", index=False)

meta_info = dict(best_config=list(best_key), base_models=[list(k) for k in base_keys], alpha=float(a_best),
                 lam=float(l_best), meta_weights=dict(zip([f"{c}" for c, _ in base_keys], meta.coef_[0].round(4).tolist())),
                 meta_bias=float(meta.intercept_[0]), tau_stacking=tau_st, tau_average=tau_avg, tau_best_single=tau_best_single)
json.dump(meta_info, open(tdir / f"{name}_stacking_info.json", "w"), indent=2)
np.savez_compressed(rdir / "final_preds.npz", stack=p_stack, avg=p_avg, best=rb["test_pred"], yte=yte,
                    tau_st=tau_st, tau_avg=tau_avg, tau_best=tau_best_single)
print(stack.round(4).to_string())
print(tests.round(4).to_string())
print(meta_info)
