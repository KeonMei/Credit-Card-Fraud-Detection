"""Statistical tests: paired stratified bootstrap (by row or by card), DeLong test, Holm correction."""
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


def _boot_weights(y, B, seed, groups=None):
    """Bootstrap resamples as per-row multiplicities.

    groups=None: rows are resampled, frauds and legitimate rows separately (stratified_boot_indices).
    groups given (card number of every row): whole cards are resampled with replacement, cards with and
    without fraud separately, and every transaction of a drawn card is taken as many times as the card."""
    n = len(y)
    if groups is None:
        for idx in stratified_boot_indices(y, B, seed):
            yield _weights(idx, n)
        return
    rng = np.random.default_rng(seed)
    codes, card = np.unique(groups, return_inverse=True)
    has_fraud = np.bincount(card, weights=(np.asarray(y) == 1).astype(float), minlength=len(codes)) > 0
    fraud_cards, clean_cards = np.where(has_fraud)[0], np.where(~has_fraud)[0]
    for _ in range(B):
        draw = np.concatenate([rng.choice(fraud_cards, len(fraud_cards)), rng.choice(clean_cards, len(clean_cards))])
        mult = np.bincount(draw, minlength=len(codes)).astype(np.float64)
        yield mult[card]


def _metric(name, y, p, thr):
    if name == "pr_auc":
        return average_precision_score(y, p)
    # confusion counts with numpy (same values as sklearn's fbeta_score / matthews_corrcoef, much faster)
    yh = p >= thr
    yb = y == 1
    tp = float(np.sum(yh & yb)); fp = float(np.sum(yh & ~yb)); fn = float(np.sum(~yh & yb)); tn = float(len(y)) - tp - fp - fn
    if name == "f2":
        return 0.0 if tp == 0 else 5 * tp / (5 * tp + 4 * fn + fp)
    if name == "mcc":
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return 0.0 if den == 0 else (tp * tn - fp * fn) / den
    raise ValueError(name)


class _Scorer:
    """Metrics for one score vector on bootstrap resamples given as per-row multiplicities (weights).
    Gives exactly the values of sklearn's average_precision_score / F2 / MCC on the resampled rows,
    but in O(n) per resample, because the sort order of the scores is computed once."""

    def __init__(self, y, p, thr):
        self.order = np.argsort(-p, kind="mergesort")
        ps = p[self.order]
        self.ys = (y[self.order] == 1).astype(np.float64)
        self.last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]  # last index of each group of equal scores
        self.flag = (p >= thr)
        self.yb = y == 1

    def metrics(self, w, names):
        out = {}
        if "pr_auc" in names:
            ws = w[self.order]
            tp = np.cumsum(ws * self.ys)[self.last]
            fp = np.cumsum(ws * (1 - self.ys))[self.last]
            prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            rec = tp / tp[-1]
            out["pr_auc"] = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
        if "roc_auc" in names:  # Mann-Whitney form with ties counted as 1/2, equal to sklearn's roc_auc_score
            ws = w[self.order]
            tp = np.cumsum(ws * self.ys)[self.last]
            fp = np.cumsum(ws * (1 - self.ys))[self.last]
            wp, wn = np.diff(np.r_[0.0, tp]), np.diff(np.r_[0.0, fp])
            neg_below = fp[-1] - fp
            out["roc_auc"] = float(np.sum(wp * (neg_below + 0.5 * wn)) / (tp[-1] * fp[-1]))
        if "f2" in names or "mcc" in names:
            tp = float(w[self.flag & self.yb].sum()); fp = float(w[self.flag & ~self.yb].sum())
            fn = float(w[~self.flag & self.yb].sum()); tn = float(w.sum()) - tp - fp - fn
            if "f2" in names:
                out["f2"] = 0.0 if tp == 0 else 5 * tp / (5 * tp + 4 * fn + fp)
            if "mcc" in names:
                den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                out["mcc"] = 0.0 if den == 0 else (tp * tn - fp * fn) / den
        return out


def _weights(idx, n):
    return np.bincount(idx, minlength=n).astype(np.float64)


def bootstrap_ci(y, p, thr, metrics=("pr_auc", "f2", "mcc"), B=2000, seed=0, groups=None):
    sc = _Scorer(y, p, thr)
    vals = {m: [] for m in metrics}
    for w in _boot_weights(y, B, seed, groups):
        r = sc.metrics(w, metrics)
        for m in metrics:
            vals[m].append(r[m])
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.std(v))) for m, v in vals.items()}


def paired_bootstrap(y, pa, ta, pb, tb, metrics=("pr_auc", "f2", "mcc"), B=2000, seed=0, groups=None):
    """Difference A - B on the same resamples (rows, or whole cards when groups is given).
    Returns mean diff, 95% CI and two-sided bootstrap p-value."""
    sa, sb = _Scorer(y, pa, ta), _Scorer(y, pb, tb)
    diffs = {m: [] for m in metrics}
    for w in _boot_weights(y, B, seed, groups):
        ra, rb = sa.metrics(w, metrics), sb.metrics(w, metrics)
        for m in metrics:
            diffs[m].append(ra[m] - rb[m])
    out = {}
    for m, v in diffs.items():
        v = np.asarray(v)
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        out[m] = dict(diff=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                      p=float(min(1.0, max(p, 1.0 / B))))
    return out
