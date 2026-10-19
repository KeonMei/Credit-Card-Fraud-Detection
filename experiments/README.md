# Experiments

Scripts for the baseline comparison of the thesis: five classifiers (LR, RF, XGBoost, LightGBM, MLP)
times four balancing settings (none, RUS, SMOTE, SMOTE-ENN) on the ULB, IEEE-CIS and Sparkov datasets,
followed by statistical tests, a stacking model with a tuned threshold and additional analyses.

| Script | What it does |
|---|---|
| `prep.py` | ULB: removes duplicate rows, time split 80/20, features; IEEE-CIS: joins the two files, time split 80/20, features from `fraud_detection.features` (fit on the training period only); ULB random split |
| `prep_ieee_random.py` | IEEE-CIS dataset statistics and the stratified random 80/20 split |
| `prep_sparkov.py` | Sparkov: joins fraudTrain/fraudTest, 25 features (amount, time, age, gender, distance, per-card activity in the previous 24 h, category), split by card (20% of cards for test, 50% of the other cards for training), 5 card-grouped folds; extra date and random splits |
| `subsample_sparkov_splits.py` | subsamples the training part of the Sparkov date and random splits to the size of the card-split training set |
| `select_features.py` | IEEE-CIS: 100 features with the highest LightGBM gain on the training period |
| `common.py` | forward-chaining folds (ULB, IEEE-CIS) or card-grouped folds (Sparkov), balancing, classifiers, search ranges, metrics, F2 threshold |
| `run_grid.py` | random search per configuration, out-of-fold predictions on 5 forward-chaining folds, refit, test predictions |
| `stats_utils.py` | paired stratified bootstrap, DeLong test, Holm correction |
| `analyze.py` | main table, best-vs-all tests, balancing-vs-none tests, metric agreement, stacking, threshold analysis |
| `sensitivity.py` | random split and "SMOTE before split" protocol compared with the time split |
| `figures.py` | PR curves, PR-AUC heatmap, threshold curves, confusion matrices, protocol comparison; error analysis table |
| `run_all.sh` | runs everything for ULB and IEEE-CIS in order (restartable) |
| `run_sparkov.sh` | runs everything for Sparkov (restartable) |

Run from the repository root after placing the Kaggle files in `data/`:

```bash
pip install -e ".[experiments]"
bash experiments/run_all.sh
bash experiments/run_sparkov.sh
```

Outputs go to `experiments/results/` (tables, figures, per-configuration pickles); intermediate arrays
go to `experiments/cache/`. Random seed 42 everywhere. Search budget per configuration
(number of random-search trials) is set in `run_all.sh`.
