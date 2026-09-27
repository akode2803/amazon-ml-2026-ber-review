#!/bin/bash
# Leave-one-country-out: current (base) vs robust (mono + regularised) stack.
# Each run trains both stages on one country and scores the other's val fold.
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 LGB_THREADS=10
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
L=$BER_ROOT/logs
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
ROB="LGB_LEAVES=63 LGB_MINLEAF=500 LGB_FF=0.5 LGB_L2=5 LGB_MONO=1 NROUND=1000 S2_LEAVES=63 S2_MINLEAF=500 S2_FF=0.5 S2_L2=5 S2_MONO=1"
wf() { WF_COUNTRIES=$2 $PY waterfall.py $1 p2 > $L/wf_$1_$2.log 2>&1; head -1 $L/wf_$1_$2.log; }

for dir in "US India" "India US"; do
  set -- $dir; tr=$1; ev=$2; t=${tr:0:2}
  TAG=loco${t}_base COUNTRIES=$tr EVAL_COUNTRIES=$ev $PY stack.py > $L/stack_loco${t}_base.log 2>&1 &
  env $ROB TAG=loco${t}_rob COUNTRIES=$tr EVAL_COUNTRIES=$ev $PY stack.py > $L/stack_loco${t}_rob.log 2>&1
  wait
  wf loco${t}_base $ev; wf loco${t}_rob $ev
  wf loco${t}_base $tr; wf loco${t}_rob $tr
done
env $ROB TAG=s2xR $PY stack.py > $L/stack_s2xR.log 2>&1
wf s2xR India; wf s2xR US
echo LOCO_DONE
