# Credit Card Fraud Detection
ML system for identifying fraudulent transactions from credit card transaction data.

MVP of the modelling part: data loading → leakage-free feature engineering → baseline and
gradient-boosting models with class-imbalance handling → evaluation with imbalance-aware
metrics → saved model that scores new transactions.

## Project structure

```
src/fraud_detection/
  data.py         loading of the Kaggle files, chronological / random split
  features.py     feature builders (fit on train only): ULB and IEEE-CIS
  models.py       logreg, random forest, XGBoost, LightGBM + weight / SMOTE / undersampling
  evaluation.py   PR-AUC, ROC-AUC, F1/F2, MCC, precision@k, fraud-amount recall, plots
  train.py        CLI: train and compare models, save best one to artifacts/<dataset>/
  predict.py      CLI + FraudScorer class: score new transactions with a saved model
tests/            pytest suite on synthetic data (no Kaggle download needed)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Data

Download from Kaggle (an account is required; for IEEE-CIS you must also accept the competition rules):

| Dataset | Link | Put files into |
|---|---|---|
| ULB Credit Card Fraud | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `data/ulb/creditcard.csv` |
| IEEE-CIS Fraud Detection | https://www.kaggle.com/competitions/ieee-fraud-detection/data | `data/ieee/train_transaction.csv`, `data/ieee/train_identity.csv` |

With the Kaggle CLI:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ulb --unzip
kaggle competitions download -c ieee-fraud-detection -p data/ieee
```

`data/` and `artifacts/` are git-ignored.

## Training

```bash
# ULB: logreg, rf, xgb, lgbm with cost-sensitive weights, chronological split
python -m fraud_detection.train --dataset ulb --data-dir data/ulb

# IEEE-CIS (needs ~6-8 GB RAM; use --nrows 200000 for a quick run)
python -m fraud_detection.train --dataset ieee --data-dir data/ieee

# experiments for the thesis
python -m fraud_detection.train --dataset ulb --data-dir data/ulb --imbalance smote --output-dir artifacts/ulb_smote
python -m fraud_detection.train --dataset ulb --data-dir data/ulb --split random --output-dir artifacts/ulb_random
python -m fraud_detection.train --dataset ieee --data-dir data/ieee --beta 2 --output-dir artifacts/ieee_recall
```

Outputs in `artifacts/<dataset>/`: `metrics.csv` (val + test for every model), `summary.json`,
`curves_test.png` (PR + ROC), confusion matrix and SHAP feature importance of the best model,
`model.joblib` (feature builder + model + threshold).

## Thesis experiments

The full baseline comparison (5 classifiers x 4 balancing methods on ULB and IEEE-CIS, statistical tests,
stacking, threshold analysis) is in [`experiments/`](experiments/README.md).

## Scoring new transactions

```bash
python -m fraud_detection.predict --model artifacts/ieee/model.joblib \
    --input data/ieee/test_transaction.csv --identity data/ieee/test_identity.csv --output scored.csv
```

```python
from fraud_detection.predict import FraudScorer
scores = FraudScorer("artifacts/ulb/model.joblib").score(transactions_df)
```

## Evaluation protocol

1. **Chronological split** 60/20/20 — train on the past, test on the future. Random splits
   (common in papers) leak future information and, in IEEE-CIS, the same client into train and test.
2. Feature statistics (frequencies, per-card aggregates, category codes) are **fitted on train only**.
3. Resampling (SMOTE / undersampling) is applied **only to the training set**.
4. Early stopping, decision threshold and model selection use **validation**; **test is used once**.
5. Primary metric is **PR-AUC**; accuracy is meaningless at 0.17% fraud.

## Roadmap (improvements grounded in the literature)

| # | Improvement | Reference |
|---|---|---|
| 1 | Delay period between train and test (labels arrive with a lag), rolling-window validation, card-level precision@k | Le Borgne et al., *Reproducible ML for Credit Card Fraud Detection — Practical Handbook*, 2022; Dal Pozzolo et al., IEEE TNNLS 2018 |
| 2 | Probability calibration after undersampling | Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015 |
| 3 | Compare resampling methods (SMOTE, ADASYN, Borderline-SMOTE) vs cost-sensitive learning | Chawla et al., JAIR 2002; He & Garcia, IEEE TKDE 2009 |
| 4 | Behavioural / aggregated features (transaction velocity, amount deviation per card over time windows) | Bahnsen et al., *Feature engineering strategies for credit card fraud detection*, Expert Systems with Applications 2016 |
| 5 | CatBoost + hyperparameter tuning with Optuna | Prokhorenkova et al., NeurIPS 2018; Akiba et al., KDD 2019 |
| 6 | Unsupervised anomaly features (Isolation Forest, autoencoder) combined with supervised model | Carcillo et al., *Combining unsupervised and supervised learning in credit card fraud detection*, Information Sciences 2021 |
| 7 | Sequence models over a card's transaction history (LSTM / Transformer) | Jurgovsky et al., *Sequence classification for credit-card fraud detection*, Expert Systems with Applications 2018 |
| 8 | Explainability for analysts: per-transaction SHAP explanations | Lundberg & Lee, NeurIPS 2017 |
| 9 | Serving: REST API (FastAPI) + drift monitoring | — |
