#!/bin/bash
# Sparkov pipeline (restartable: finished configurations and finished search trials are skipped).
# Expects data/sparkov/fraudTrain.csv and data/sparkov/fraudTest.csv. Run after run_all.sh
# (figures.py and make_tables.py collect all three datasets).
set -e
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONWARNINGS=ignore
[ -f cache/sparkov_card.npz ] || { python3 prep_sparkov.py && python3 subsample_sparkov_splits.py; } >> logs/sparkov_grid.log 2>&1
python3 run_grid.py sparkov_card "LR:6,RF:4,XGB:6,LGBM:6,MLP:4" 1 >> logs/sparkov_grid.log 2>&1
python3 analyze.py sparkov_card 2000 > logs/analyze_sparkov.log 2>&1
python3 sensitivity.py sparkov > logs/sensitivity_sparkov.log 2>&1
python3 figures.py > logs/figures.log 2>&1
python3 make_tables.py > logs/tables.log 2>&1
