#!/bin/bash
# Sequential queue (two US-sized stacks in parallel exceed 31 GB RAM).
# Priority: LOCO US->India base/rob, self-training proxy, LOCO India->US, joint robust.
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 LGB_THREADS=20
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
L=$BER_ROOT/logs
D=$BER_ROOT/work/dumps
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
ROB="LGB_LEAVES=63 LGB_MINLEAF=500 LGB_FF=0.5 LGB_L2=5 LGB_MONO=1 NROUND=1000 S2_LEAVES=63 S2_MINLEAF=500 S2_FF=0.5 S2_L2=5 S2_MONO=1"
wf() { WF_COUNTRIES=$2 $PY waterfall.py $1 p2 > $L/wf_$1_$2.log 2>&1; echo "RESULT $(head -1 $L/wf_$1_$2.log)"; }

# locoUS_rob done: India 0.87499, US 0.98492
TAG=locoUS_base COUNTRIES=US EVAL_COUNTRIES=India $PY stack.py > $L/stack_locoUS_base.log 2>&1
wf locoUS_base India; wf locoUS_base US

SPLIT=train TAG=locoUS_base COUNTRIES=India DUMP=$D/ps_IN_from_US.npz $PY infer2.py > $L/infer_ps_IN.log 2>&1
TAG=locoUS_ps COUNTRIES=US EVAL_COUNTRIES=India \
  PSEUDO="India|train|$D/ps_IN_from_US.npz|0.98|0.02" $PY stack.py > $L/stack_locoUS_ps.log 2>&1
wf locoUS_ps India

TAG=locoIN_base COUNTRIES=India EVAL_COUNTRIES=US $PY stack.py > $L/stack_locoIN_base.log 2>&1
wf locoIN_base US; wf locoIN_base India
env $ROB TAG=locoIN_rob COUNTRIES=India EVAL_COUNTRIES=US $PY stack.py > $L/stack_locoIN_rob.log 2>&1
wf locoIN_rob US; wf locoIN_rob India

env $ROB TAG=s2xR $PY stack.py > $L/stack_s2xR.log 2>&1
wf s2xR India; wf s2xR US
echo SEQ_DONE
