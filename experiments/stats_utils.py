"""Statistical tests: paired stratified bootstrap, DeLong test, Holm correction."""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def delong_test(y, p1, p2):
    """DeLong et al. (1988) test for two correlated ROC AUCs (fast version, Sun & Xu 2014).
    Returns auc1, auc2, z, two-sided p-value."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([p1, p2])[:, order]
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def stratified_boot_indices(y, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(B):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def _metric(name, y, p, thr):
    if name == "pr_auc":
        return average_precision_score(y, p)
    yhat = (p >= thr).astype(int)
    if name == "f2":
        return fbeta_score(y, yhat, beta=2, zero_division=0)
    if name == "mcc":
        return matthews_corrcoef(y, yhat)
    raise ValueError(name)


def bootstrap_ci(y, p, thr, metrics=("pr_auc", "f2", "mcc"), B=2000, seed=0):
    vals = {m: [] for m in metrics}
    for idx in stratified_boot_indices(y, B, seed):
        for m in metrics:
            vals[m].append(_metric(m, y[idx], p[idx], thr))
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.std(v))) for m, v in vals.items()}


def paired_bootstrap(y, pa, ta, pb, tb, metrics=("pr_auc", "f2", "mcc"), B=2000, seed=0):
    """Difference A - B on the same resamples. Returns mean diff, 95% CI and two-sided bootstrap p-value."""
    diffs = {m: [] for m in metrics}
    for idx in stratified_boot_indices(y, B, seed):
        yy = y[idx]
        for m in metrics:
            diffs[m].append(_metric(m, yy, pa[idx], ta) - _metric(m, yy, pb[idx], tb))
    out = {}
    for m, v in diffs.items():
        v = np.asarray(v)
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        out[m] = dict(diff=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                      p=float(min(1.0, max(p, 1.0 / B))))
    return out
