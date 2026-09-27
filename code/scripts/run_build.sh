#!/bin/bash
# v2 cache build (phonetic skeleton rules on) into ber2/work.
set -e
export BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 BER_SKEL2=1 PYTHONIOENCODING=utf-8
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
L=/c/Users/LENOVO/Desktop/Amazon_ML/ber2/logs
t0=$(date +%s); step() { echo "=== $* [$(( $(date +%s) - t0 ))s]"; }
step prep;  $PY prep.py train test
step terms wave 1
$PY terms.py train US > $L/t_train_US.log 2>&1 &
$PY terms.py train India > $L/t_train_India.log 2>&1 &
wait
step terms wave 2
$PY terms.py test US > $L/t_test_US.log 2>&1 &
$PY terms.py test India > $L/t_test_India.log 2>&1 &
$PY terms.py test France > $L/t_test_France.log 2>&1 &
wait
cat $L/t_*.log
step sigs;  $PY sigs.py train; $PY sigs.py test
step cand train; $PY candgen.py --split train --k-fwd 20 --k-rev 5 --out c1
step cand test;  $PY candgen.py --split test  --k-fwd 20 --k-rev 5 --out c1
step done
