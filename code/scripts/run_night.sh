#!/bin/bash
# 1) v5b = self-trained France with "explained" negatives (India proxy 0.90427)
# 2) round-2 self-training proxy: pseudo-label India again with locoUS_ps2
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 LGB_THREADS=20
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
L=$BER_ROOT/logs
D=$BER_ROOT/work/dumps
cd $BER_ROOT
PSEUDO_NEG=explained STAG=s2xP2 OUT=fr_p2 V=v5b bash run_v5.sh > $L/run_v5b.log 2>&1
tail -n 3 $L/run_v5b.log

cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
SPLIT=train TAG=locoUS_ps2 COUNTRIES=India DUMP=$D/ps2_IN.npz $PY infer2.py > $L/infer_ps2_IN.log 2>&1
TAG=locoUS_ps3 COUNTRIES=US EVAL_COUNTRIES=India PSEUDO_NEG=explained \
  PSEUDO="India|train|$D/ps2_IN.npz|0.98|0.02" $PY stack.py > $L/stack_locoUS_ps3.log 2>&1
WF_COUNTRIES=India $PY waterfall.py locoUS_ps3 p2 > $L/wf_locoUS_ps3_India.log 2>&1
echo "RESULT $(head -1 $L/wf_locoUS_ps3_India.log)"
echo NIGHT_DONE
