#!/bin/bash
# v5 = v4 with a self-trained France model.  The joint stack (France's zero-shot
# model) is retrained with France test pairs pseudo-labelled by v4's own p2
# added to stage 1; US/India keep their v4 expert dumps unchanged.
set -e
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 LGB_THREADS=20
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
HI=${HI:-0.98}; LO=${LO:-0.02}
# v5: STAG=s2xP SRC=fr_x OUT=fr_p V=v5 ; v5b: PSEUDO_NEG=explained STAG=s2xP2 OUT=fr_p2 V=v5b
STAG=${STAG:-s2xP}; SRC=${SRC:-fr_x}; OUT=${OUT:-fr_p}; V=${V:-v5}
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
TAG=$STAG PSEUDO="France|test|$D/$SRC.npz|$HI|$LO" $PY stack.py > $L/stack_$STAG.log 2>&1
WF_COUNTRIES=India $PY waterfall.py $STAG p2 | head -1
WF_COUNTRIES=US    $PY waterfall.py $STAG p2 | head -1
TAG=$STAG COUNTRIES=France DUMP=$D/$OUT.npz $PY infer2.py > $L/infer_$V.log 2>&1
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$BER_ROOT/output_$V \
  $PY assemble.py $D/$OUT.npz $D/us_x.npz $D/in_x.npz
cd $BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource
$PY utils/validate_submission.py --matching $BER_ROOT/output_$V/matching_results.tsv \
  --candidate $BER_ROOT/output_$V/candidate_pairs.tsv --test-dir dataset/test
echo ${V}_DONE
