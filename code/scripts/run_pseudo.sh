#!/bin/bash
# Self-training proxy for France: US-only stack pseudo-labels India (all folds,
# labels unseen), retrain with those rows, score India's val fold for real.
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 LGB_THREADS=10
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
L=$BER_ROOT/logs
D=$BER_ROOT/work/dumps
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
until grep -q "saved val scores" $L/stack_locoUS_base.log; do sleep 20; done
SPLIT=train TAG=locoUS_base COUNTRIES=India DUMP=$D/ps_IN_from_US.npz $PY infer2.py > $L/infer_ps_IN.log 2>&1
TAG=locoUS_ps COUNTRIES=US EVAL_COUNTRIES=India \
  PSEUDO="India|train|$D/ps_IN_from_US.npz|0.98|0.02" $PY stack.py > $L/stack_locoUS_ps.log 2>&1
WF_COUNTRIES=India $PY waterfall.py locoUS_ps p2 > $L/wf_locoUS_ps_India.log 2>&1
head -1 $L/wf_locoUS_ps_India.log
echo PSEUDO_DONE
