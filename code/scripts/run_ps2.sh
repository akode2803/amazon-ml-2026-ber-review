#!/bin/bash
# Self-training proxy, variant: "explained" negatives only (see stack.py
# PSEUDO_NEG).  Compare with locoUS_ps (all negatives) = India 0.89508.
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 LGB_THREADS=20
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
L=$BER_ROOT/logs
D=$BER_ROOT/work/dumps
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
while ps -ef | grep -v grep | grep -q "run_v5.sh"; do sleep 15; done   # one job at a time
TAG=locoUS_ps2 COUNTRIES=US EVAL_COUNTRIES=India PSEUDO_NEG=explained \
  PSEUDO="India|train|$D/ps_IN_from_US.npz|0.98|0.02" $PY stack.py > $L/stack_locoUS_ps2.log 2>&1
WF_COUNTRIES=India $PY waterfall.py locoUS_ps2 p2 > $L/wf_locoUS_ps2_India.log 2>&1
echo "RESULT $(head -1 $L/wf_locoUS_ps2_India.log)"
echo PS2_DONE
