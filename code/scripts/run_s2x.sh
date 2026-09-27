#!/bin/bash
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
L=/c/Users/LENOVO/Desktop/Amazon_ML/ber2/logs
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
TAG=s2x $PY stack.py > $L/stack_s2x.log 2>&1 &
TAG=s2xIN COUNTRIES=India $PY stack.py > $L/stack_s2xIN.log 2>&1
wait
TAG=s2xUS COUNTRIES=US $PY stack.py > $L/stack_s2xUS.log 2>&1
echo S2X_DONE
