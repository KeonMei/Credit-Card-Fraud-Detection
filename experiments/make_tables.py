"""Collects all result tables into one Excel workbook and LaTeX table snippets (numbers only, no prose)."""
import json
import pickle

import numpy as np
import pandas as pd

from common import BALANCING, CLASSIFIERS, RES

T = RES / "tables"
OUT = RES / "latex"
OUT.mkdir(exist_ok=True)
info = json.load(open(RES.parent / "cache" / "data_info.json"))
DS = {"ulb": "ULB", "ieee": "IEEE-CIS", "sparkov": "Sparkov"}
MAIN = {"ulb": "ulb_time", "ieee": "ieee_time", "sparkov": "sparkov_card"}
DS = {k: v for k, v in DS.items() if (T / f"{MAIN[k]}_main.csv").exists()}


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


sheets = {}

# ---------------------------------------------------------------- data table
u, i = info["ulb"], info["ieee"]
recs = [
    dict(dataset="ULB (European cardholders)", source="Kaggle mlg-ulb/creditcardfraud", period="2 days, Sept. 2013",
         raw_rows=u["raw_rows"], removed=u["duplicates_removed"], removed_reason="exact duplicate rows",
         rows_after_cleaning=u["rows"], frauds=u["frauds"], fraud_rate=u["frauds"] / u["rows"],
         split="time 80/20", train_rows=u["train_rows"], train_frauds=u["train_frauds"],
         test_rows=u["test_rows"], test_frauds=u["test_frauds"],
         train_period=f"0-{u['train_time_range_h'][1]:.1f} h", test_period=f"{u['test_time_range_h'][0]:.1f}-{u['test_time_range_h'][1]:.1f} h",
         features=str(u["n_features"])),
    dict(dataset="IEEE-CIS (Vesta)", source="Kaggle ieee-fraud-detection (train files)", period="183 days (TransactionDT)",
         raw_rows=i["raw_rows"], removed=0, removed_reason=f"none ({i['duplicate_rows_excluding_id']} rows identical except ID kept)",
         rows_after_cleaning=i["rows"], frauds=i["frauds"], fraud_rate=i["frauds"] / i["rows"],
         split="time 80/20", train_rows=i["train_rows"], train_frauds=i["train_frauds"], test_rows=i["test_rows"], test_frauds=i["test_frauds"],
         train_period=f"day {i['train_day_range'][0]:.0f}-{i['train_day_range'][1]:.1f}",
         test_period=f"day {i['test_day_range'][0]:.1f}-{i['test_day_range'][1]:.1f}", features="100 of 454"),
]
if "sparkov" in info and "sparkov" in DS:
    sp = info["sparkov"]
    recs.append(dict(dataset="Sparkov (simulated)", source="Kaggle kartik2112/fraud-detection (Sparkov generator)",
                     period="Jan 2019 - Dec 2020", raw_rows=sp["raw_rows"], removed=sp["duplicates_removed"],
                     removed_reason="none (no duplicate transaction ids)", rows_after_cleaning=sp["rows"], frauds=sp["frauds"],
                     fraud_rate=sp["frauds"] / sp["rows"],
                     split=f"by card: {sp['test_cards']} test cards; {sp['train_cards']} of {sp['train_cards_available']} other cards for training",
                     train_rows=sp["train_rows"], train_frauds=sp["train_frauds"], test_rows=sp["test_rows"], test_frauds=sp["test_frauds"],
                     train_period="whole period", test_period="whole period", features=str(sp["n_features"])))
data = pd.DataFrame(recs)
sheets["data"] = data
cols = list(range(len(data)))
names = ["ULB", "IEEE-CIS", "Sparkov"][:len(data)]
with open(OUT / "tab_data.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "r" * len(data) + "}\n\\hline\n & " + " & ".join(names) + " \\\\\n\\hline\n")
    def rate(r):
        return f"{r.frauds:,} ({100 * r.fraud_rate:.3f}\\%)"
    rows = [("Period", *[data.period[c] for c in cols]),
            ("Records (raw)", *[f"{data.raw_rows[c]:,}" for c in cols]),
            ("Removed (duplicates)", *[f"{data.removed[c]:,}" for c in cols]),
            ("Records after cleaning", *[f"{data.rows_after_cleaning[c]:,}" for c in cols]),
            ("Frauds (rate)", *[rate(data.iloc[c]) for c in cols]),
            ("Split", *["time 80/20", "time 80/20", "by card"][:len(data)]),
            ("Train records / frauds", *[f"{data.train_rows[c]:,} / {data.train_frauds[c]:,}" for c in cols]),
            ("Test records / frauds", *[f"{data.test_rows[c]:,} / {data.test_frauds[c]:,}" for c in cols]),
            ("Train period", *[data.train_period[c] for c in cols]),
            ("Test period", *[data.test_period[c] for c in cols]),
            ("Features", *[data.features[c] for c in cols])]
    for r in rows:
        f.write(" & ".join(r).replace(",", "{,}") + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

# ---------------------------------------------------------------- main tables
for ds, label in DS.items():
    m = pd.read_csv(T / f"{MAIN[ds]}_main.csv")
    m = m.set_index(["classifier", "balancing"]).reindex(pd.MultiIndex.from_product([CLASSIFIERS, BALANCING])).reset_index()
    m = m.rename(columns={"level_0": "classifier", "level_1": "balancing"})
    sheets[f"{ds}_main"] = m
    best = m.loc[m.cv_pr_auc_mean.idxmax()]
    with open(OUT / f"tab_main_{ds}.tex", "w") as f:
        f.write("\\begin{tabular}{llcccccc}\n\\hline\nModel & Balancing & CV PR-AUC & Test PR-AUC [95\\% CI] & ROC-AUC & $F_2$ & MCC & FPR (\\%) \\\\\n\\hline\n")
        for _, r in m.iterrows():
            mark = "\\textbf{" if (r.classifier == best.classifier and r.balancing == best.balancing) else "{"
            f.write(f"{r.classifier} & {r.balancing} & {mark}{r.cv_pr_auc_mean:.3f}}} $\\pm$ {r.cv_pr_auc_std:.3f} & "
                    f"{r.test_pr_auc:.3f} [{r.pr_auc_lo:.3f}, {r.pr_auc_hi:.3f}] & {r.test_roc_auc:.3f} & "
                    f"{r.f2_tau:.3f} & {r.mcc_tau:.3f} & {100*r.fpr_tau:.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # stacking
    s = pd.read_csv(T / f"{MAIN[ds]}_stacking_vs_single.csv")
    st = pd.read_csv(T / f"{MAIN[ds]}_stacking_tests.csv")
    sinfo = json.load(open(T / f"{MAIN[ds]}_stacking_info.json"))
    sheets[f"{ds}_stacking"] = s
    sheets[f"{ds}_stacking_tests"] = st
    rf = pd.read_csv(T / f"{MAIN[ds]}_recall_at_fpr.csv")
    sheets[f"{ds}_recall_at_fpr"] = rf
    with open(OUT / f"tab_stacking_{ds}.tex", "w") as f:
        f.write("\\begin{tabular}{llccccccc}\n\\hline\nModel & $\\tau$ & PR-AUC & Precision & Recall & $F_2$ [95\\% CI] & MCC & FPR (\\%) & FP / FN \\\\\n\\hline\n")
        for _, r in s.iterrows():
            f.write(f"{r.model} & {r.threshold:.3f} & {r.pr_auc:.3f} & {r.precision:.3f} & {r.recall:.3f} & "
                    f"{r.f2:.3f} [{r.f2_lo:.3f}, {r.f2_hi:.3f}] & {r.mcc:.3f} & {100*r.fpr:.3f} & {int(r.fp)} / {int(r.fn)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # best vs all and RQ1
    c = pd.read_csv(T / f"{MAIN[ds]}_best_vs_all.csv")
    q = pd.read_csv(T / f"{MAIN[ds]}_rq1_balancing_vs_none.csv")
    sheets[f"{ds}_best_vs_all"] = c
    sheets[f"{ds}_rq1_bal_vs_none"] = q
    sheets[f"{ds}_rq2_rank_corr"] = pd.read_csv(T / f"{MAIN[ds]}_rq2_metric_rank_correlation.csv")
    sheets[f"{ds}_stacking_grid"] = pd.read_csv(T / f"{MAIN[ds]}_stacking_grid.csv")
    with open(OUT / f"tab_rq1_{ds}.tex", "w") as f:
        f.write("\\begin{tabular}{llccc}\n\\hline\nModel & Balancing vs none & $\\Delta$PR-AUC [95\\% CI] & $p_{\\mathrm{Holm}}$ (PR-AUC) & $p_{\\mathrm{Holm}}$ (DeLong) \\\\\n\\hline\n")
        for _, r in q.iterrows():
            f.write(f"{r.classifier} & {r.balancing} & {r.d_pr_auc:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] & {pval(r.p_pr_auc_holm)} & {pval(r.p_delong_holm)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    sens = T / f"{ds}_sensitivity.csv"
    if sens.exists():
        sheets[f"{ds}_sensitivity"] = pd.read_csv(sens)

err = T / "error_analysis.csv"
if err.exists():
    sheets["error_analysis"] = pd.read_csv(err)

# protocol table across datasets
rows = []
for ds in DS:
    f = T / f"{ds}_sensitivity.csv"
    if f.exists():
        rows.append(pd.read_csv(f))
if rows:
    sp = pd.concat(rows)
    with open(OUT / "tab_protocols.tex", "w") as f:
        f.write("\\begin{tabular}{lllccccc}\n\\hline\nData & Model & Protocol & PR-AUC & ROC-AUC & Precision & Recall & MCC \\\\\n\\hline\n")
        for _, r in sp.iterrows():
            f.write(f"{DS[r.dataset]} & {r.config} & {r.protocol} & {r.pr_auc:.3f} & {r.roc_auc:.3f} & {r.precision:.3f} & {r.recall:.3f} & {r.mcc:.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

# runtime per configuration
rt = []
for ds in DS:
    for c in CLASSIFIERS:
        for b in BALANCING:
            p = RES / MAIN[ds] / f"{c}__{b}.pkl"
            if p.exists():
                r = pickle.load(open(p, "rb"))
                rt.append(dict(dataset=ds, classifier=c, balancing=b, n_trials=len(r["trials"]), search_seconds=r["search_seconds"],
                               final_fit_seconds=r["fit_seconds_full"], train_rows_after_balancing=r["n_fit"],
                               train_frauds_after_balancing=r["n_pos_fit"], ratio=r["ratio"], params=json.dumps(r["params"])))
sheets["runtime_and_params"] = pd.DataFrame(rt)

with pd.ExcelWriter(RES / "results_all.xlsx") as xw:
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name[:31], index=False)
print("tables done:", list(sheets))
