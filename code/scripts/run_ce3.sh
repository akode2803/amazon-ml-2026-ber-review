#!/bin/bash
# CE3 = cross-encoder self-trained on the test set of every country: train pairs
# plus test pairs pseudo-labelled by the best blended dumps (v9, CE on all
# pairs), so the text model sees the test set's own distractors.
#   v10  = v7 + mean(CE1, CE2, CE3) on all kept pairs
#   v10b = v10 with the CE coefficient boosted 1.5x
# Starts only after run_phase2.sh is done (one GPU job at a time).
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 HF_HUB_OFFLINE=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
VAL="$BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource"
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0"
MODELS="s2xP3:fr_p3:US,India s2xUS_P2:us_p2:US s2xIN_P2:in_p2:India"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
until grep -q "output_v9b_DONE" $L/run_phase2.log 2>/dev/null; do
  ps -ef | grep -v grep | grep -q "run_phase2.sh" || { echo "phase2 ended without v9b"; exit 1; }
  sleep 20
done
while ps -ef | grep -v grep | grep -q "xenc.py"; do sleep 10; done

VAL_TAG=s2xP3 DUMPS=fr_p3 MODEL_TAG=3 N_TRAIN=2000000 \
  PSEUDO_DUMP="France:fr_p3ce_all:300000;US:us_p2ce_all:300000;India:in_p2ce_all:300000" \
  $PY xenc.py data > $L/xenc3_data.log 2>&1
grep -E "pseudo" $L/xenc3_data.log
MODEL_TAG=3 $PY xenc.py train > $L/xenc3_train.log 2>&1
grep -E "holdout" $L/xenc3_train.log
for m in $MODELS; do
  IFS=: read T d E <<< "$m"
  VAL_TAG=$T DUMPS=$d MODEL_TAG=3 BAND_TAG=_all BAND_HI=1.01 $PY xenc.py score > $L/ce3_s_$d.log 2>&1
done
echo CE3_SCORED

for spec in "_all3:1:output_v10" "_all3b:1.5:output_v10b"; do
  IFS=: read ot boost od <<< "$spec"
  for m in $MODELS; do
    IFS=: read T d E <<< "$m"
    echo "== $T out'$ot' boost $boost"
    VAL_TAG=$T DUMPS=$d EVAL_CN=$E CE_TAGS=",2,3" BAND_TAG=_all BAND_HI=1.01 OUT_TAG=$ot \
      CE_BOOST=$boost $PY xenc.py blend > $L/ce3_b_${d}$ot.log 2>&1
    grep -E "blend coef| (US|India) " $L/ce3_b_${d}$ot.log
  done
  PARAMS=$PARAMS OUTDIR=$BER_ROOT/$od $PY assemble.py $D/fr_p3ce$ot.npz $D/us_p2ce$ot.npz \
    $D/in_p2ce$ot.npz > $L/assemble_$od.log 2>&1
  grep "decision:" $L/assemble_$od.log
  ( cd $VAL && $PY utils/validate_submission.py --matching $BER_ROOT/$od/matching_results.tsv \
      --candidate $BER_ROOT/$od/candidate_pairs.tsv --test-dir dataset/test > $L/validate_$od.log 2>&1 )
  tail -n 1 $L/validate_$od.log; echo "${od}_DONE"
done
