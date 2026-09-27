#!/bin/bash
# v12 = v7 stacks + CE1..CE4 (local, e5-small) + CE5 (RunPod, e5-large), all pairs.
#   output_v12    mean of the five CE logits, boost 1.5   (v10b recipe)
#   output_v12s   one blend weight per CE (CE_SEP), boost 1.5
#   output_v12s1  one blend weight per CE, boost 1.0
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 HF_HUB_OFFLINE=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
VAL="$BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource"
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0"
MODELS="s2xP3:fr_p3:US,India s2xUS_P2:us_p2:US s2xIN_P2:in_p2:India"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
until [ -f $BER_ROOT/work/runpod/logits_ce5.npy ] && ! ps -ef | grep -v grep | grep -q "scp .*logits_ce5"; do sleep 20; done
sleep 5
$PY import_ce.py C:/Users/LENOVO/Desktop/Amazon_ML/ber2/work/runpod/logits_ce5.npy 5
for spec in "_all5b:1.5:0:output_v12" "_all5sb:1.5:1:output_v12s" "_all5s:1.0:1:output_v12s1"; do
  IFS=: read ot boost sep od <<< "$spec"
  for m in $MODELS; do
    IFS=: read T d E <<< "$m"
    echo "== $od $T"
    VAL_TAG=$T DUMPS=$d EVAL_CN=$E CE_TAGS=",2,3,4,5" CE_SEP=$sep BAND_TAG=_all BAND_HI=1.01 \
      OUT_TAG=$ot CE_BOOST=$boost $PY xenc.py blend > $L/v12_b_${d}$ot.log 2>&1
    grep -E "blend coef|ce_all5" $L/v12_b_${d}$ot.log | grep -v wrote
  done
  PARAMS=$PARAMS OUTDIR=$BER_ROOT/$od $PY assemble.py $D/fr_p3ce$ot.npz $D/us_p2ce$ot.npz \
    $D/in_p2ce$ot.npz > $L/assemble_$od.log 2>&1
  grep "decision:" $L/assemble_$od.log
  ( cd $VAL && $PY utils/validate_submission.py --matching $BER_ROOT/$od/matching_results.tsv \
      --candidate $BER_ROOT/$od/candidate_pairs.tsv --test-dir dataset/test > $L/validate_$od.log 2>&1 )
  tail -n 1 $L/validate_$od.log; echo "${od}_DONE"
done
echo V12_ALL_DONE
