#!/bin/bash
# CE4 = next self-training round of the text model: train pairs (new sample,
# SEED=21) + test pairs of all three countries pseudo-labelled by v10b's
# blended dumps (CE1+2+3, all pairs, boost 1.5).
#   v11  = v7 + mean(CE1..CE4) on all kept pairs, boost 1.5
#   v11a = same, boost 1.0
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 HF_HUB_OFFLINE=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
VAL="$BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource"
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0"
MODELS="s2xP3:fr_p3:US,India s2xUS_P2:us_p2:US s2xIN_P2:in_p2:India"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
while ps -ef | grep -v grep | grep -q "xenc.py"; do sleep 10; done   # one GPU job at a time

VAL_TAG=s2xP3 DUMPS=fr_p3 MODEL_TAG=4 SEED=21 N_TRAIN=2000000 \
  PSEUDO_DUMP="France:fr_p3ce_all3b:400000;US:us_p2ce_all3b:400000;India:in_p2ce_all3b:400000" \
  $PY xenc.py data > $L/xenc4_data.log 2>&1
grep -E "pseudo" $L/xenc4_data.log
MODEL_TAG=4 SEED=21 $PY xenc.py train > $L/xenc4_train.log 2>&1
grep -E "holdout" $L/xenc4_train.log
for m in $MODELS; do
  IFS=: read T d E <<< "$m"
  VAL_TAG=$T DUMPS=$d MODEL_TAG=4 BAND_TAG=_all BAND_HI=1.01 $PY xenc.py score > $L/ce4_s_$d.log 2>&1
done
echo CE4_SCORED

for spec in "_all4b:1.5:output_v11" "_all4:1.0:output_v11a"; do
  IFS=: read ot boost od <<< "$spec"
  for m in $MODELS; do
    IFS=: read T d E <<< "$m"
    echo "== $T out'$ot' boost $boost"
    VAL_TAG=$T DUMPS=$d EVAL_CN=$E CE_TAGS=",2,3,4" BAND_TAG=_all BAND_HI=1.01 OUT_TAG=$ot \
      CE_BOOST=$boost $PY xenc.py blend > $L/ce4_b_${d}$ot.log 2>&1
    grep -E "blend coef|ce_all4" $L/ce4_b_${d}$ot.log | grep -v wrote
  done
  PARAMS=$PARAMS OUTDIR=$BER_ROOT/$od $PY assemble.py $D/fr_p3ce$ot.npz $D/us_p2ce$ot.npz \
    $D/in_p2ce$ot.npz > $L/assemble_$od.log 2>&1
  grep "decision:" $L/assemble_$od.log
  ( cd $VAL && $PY utils/validate_submission.py --matching $BER_ROOT/$od/matching_results.tsv \
      --candidate $BER_ROOT/$od/candidate_pairs.tsv --test-dir dataset/test > $L/validate_$od.log 2>&1 )
  tail -n 1 $L/validate_$od.log; echo "${od}_DONE"
done
