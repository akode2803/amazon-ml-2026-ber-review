#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps; mkdir -p $D
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
TAG=s2c2   COUNTRIES=France DUMP=$D/fr.npz $PY infer2.py
TAG=s2c2US COUNTRIES=US     DUMP=$D/us.npz $PY infer2.py
TAG=s2c2IN COUNTRIES=India  DUMP=$D/in.npz $PY infer2.py
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$BER_ROOT/output $PY assemble.py $D/fr.npz $D/us.npz $D/in.npz
cd $BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource
$PY utils/validate_submission.py --matching $BER_ROOT/output/matching_results.tsv --candidate $BER_ROOT/output/candidate_pairs.tsv --test-dir dataset/test
echo FINAL_DONE
