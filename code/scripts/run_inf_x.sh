#!/bin/bash
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
TAG=s2x   COUNTRIES=France DUMP=$D/fr_x.npz $PY infer2.py
TAG=s2xIN COUNTRIES=India  DUMP=$D/in_x.npz $PY infer2.py
echo INF_X_DONE
