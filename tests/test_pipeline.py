import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import SPECS, load_dataset, split_data
from fraud_detection.evaluation import compute_metrics, find_best_threshold, precision_at_k
from fraud_detection.features import IEEEFeatureBuilder
from fraud_detection.models import FraudModel
from fraud_detection.predict import FraudScorer
from fraud_detection.train import main as train_main


def test_time_split_is_chronological(ulb_dir):
    spec = SPECS["ulb"]
    train, val, test = split_data(load_dataset("ulb", ulb_dir), spec, "time")
    assert train[spec.time_col].max() <= val[spec.time_col].min()
    assert val[spec.time_col].max() <= test[spec.time_col].min()
    assert len(train) + len(val) + len(test) == 6000


def test_ieee_features_are_fitted_on_train_only(ieee_dir):
    df = load_dataset("ieee", ieee_dir)
    train, val, _ = split_data(df, SPECS["ieee"], "time")
    builder = IEEEFeatureBuilder().fit(train)
    X_val = builder.transform(val)

    assert list(X_val.columns) == builder.feature_names_
    assert "uid_freq" in X_val and "amount_mean_by_uid" in X_val
    assert not any(pd.api.types.is_object_dtype(X_val[c]) or pd.api.types.is_string_dtype(X_val[c]) for c in X_val)
    # a card never seen in training must get frequency 0, not a value computed on validation
    unseen = val.assign(card1=-1, addr1=-1.0)
    assert (builder.transform(unseen)["card1_addr1_freq"] == 0).all()


def test_threshold_and_metrics():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    threshold = find_best_threshold(y, p)
    assert threshold == pytest.approx(0.7)
    m = compute_metrics(y, p, threshold, amount=np.array([1, 1, 1, 1, 10, 30]), k=2)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["fraud_amount_recall"] == 1.0
    assert precision_at_k(y, p, 3) == pytest.approx(2 / 3)


@pytest.mark.parametrize("name", ["logreg", "rf", "xgb", "lgbm"])
@pytest.mark.parametrize("imbalance", ["weight", "smote", "undersample"])
def test_models_learn_signal(ulb_dir, name, imbalance):
    spec = SPECS["ulb"]
    train, val, test = split_data(load_dataset("ulb", ulb_dir), spec, "time")
    features = [c for c in train.columns if c != spec.target]
    model = FraudModel(name, imbalance=imbalance, n_jobs=1)
    model.fit(train[features], train[spec.target], val[features], val[spec.target])
    proba = model.predict_proba(test[features])
    assert compute_metrics(test[spec.target], proba, 0.5)["roc_auc"] > 0.9


@pytest.mark.parametrize("dataset", ["ulb", "ieee"])
def test_end_to_end_train_and_predict(dataset, request, tmp_path):
    data_dir = request.getfixturevalue(f"{dataset}_dir")
    out_dir = tmp_path / "artifacts"
    train_main(["--dataset", dataset, "--data-dir", str(data_dir), "--output-dir", str(out_dir),
                "--models", "logreg", "lgbm", "--top-k", "20"])

    for name in ["metrics.csv", "summary.json", "model.joblib", "curves_test.png"]:
        assert (out_dir / name).exists()
    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert set(metrics["model"]) == {"logreg", "lgbm"}

    raw = load_dataset(dataset, data_dir).head(50).drop(columns=[SPECS[dataset].target])
    scored = FraudScorer(out_dir / "model.joblib").score(raw)
    assert len(scored) == 50 and scored["fraud_probability"].between(0, 1).all()
