#!/bin/bash
# Full experiment pipeline. Every step skips work that is already saved, so it can be restarted.
# Expects data/ulb/creditcard.csv and data/ieee/train_{transaction,identity}.csv (see the main README).
set -e
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONWARNINGS=ignore
[ -f cache/ulb_time.npz ] && [ -f cache/ieee_time.npz ] || python3 prep.py
[ -f cache/ieee_random.npz ] || python3 prep_ieee_random.py
python3 run_grid.py ulb_time "LR:10,RF:5,XGB:10,LGBM:10,MLP:6" 1 >> logs/ulb_grid.log 2>&1
[ -f results/ieee_time/selected_features.npy ] || python3 select_features.py >> logs/ieee_grid.log 2>&1
python3 run_grid.py ieee_time "LR:6,RF:4,XGB:6,LGBM:6,MLP:4" 1 >> logs/ieee_grid.log 2>&1
python3 analyze.py ulb_time 2000 > logs/analyze_ulb.log 2>&1
python3 analyze.py ieee_time 2000 > logs/analyze_ieee.log 2>&1
python3 sensitivity.py ulb > logs/sensitivity_ulb.log 2>&1
python3 sensitivity.py ieee > logs/sensitivity_ieee.log 2>&1
python3 figures.py
python3 make_tables.py
