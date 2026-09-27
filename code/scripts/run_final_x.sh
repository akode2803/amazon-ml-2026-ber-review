#!/bin/bash
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
until grep -q INF_X_DONE $BER_ROOT/logs/inf_x.log; do sleep 10; done
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
TAG=s2xUS COUNTRIES=US DUMP=$D/us_x.npz $PY infer2.py
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$BER_ROOT/output_x $PY assemble.py $D/fr_x.npz $D/us_x.npz $D/in_x.npz
cd $BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource
$PY utils/validate_submission.py --matching $BER_ROOT/output_x/matching_results.tsv --candidate $BER_ROOT/output_x/candidate_pairs.tsv --test-dir dataset/test
echo FINAL_X_DONE
