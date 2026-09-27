#!/bin/bash
# v8 = v7 (round-2 self-training) + cross-encoder blend, averaging CE1
# (train pairs) and CE2 (train + France pseudo pairs).  Each blend is fitted
# on the val file of the model whose p2 it re-scores.  The CE1-only blend is
# printed first so the value of CE2 on val is visible.
set -e
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 HF_HUB_OFFLINE=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
until grep -q "V7_DONE" $L/run_v7.log 2>/dev/null; do sleep 20; done
until grep -qE "train done|Traceback" $L/xenc2_train.log 2>/dev/null; do sleep 20; done
grep -q "train done" $L/xenc2_train.log
while ps -ef | grep -v grep | grep -q "run_v6ce.sh"; do sleep 15; done   # GPU one job at a time
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
for spec in "France:s2xP3:fr_p3:US,India" "US:s2xUS_P2:us_p2:US" "India:s2xIN_P2:in_p2:India"; do
  IFS=: read C T d E <<< "$spec"
  export VAL_TAG=$T DUMPS=$d EVAL_CN=$E
  MODEL_TAG= $PY xenc.py data > $L/xenc_v8_data_$C.log 2>&1
  MODEL_TAG=  $PY xenc.py score > $L/xenc_v8_score1_$C.log 2>&1
  MODEL_TAG=2 $PY xenc.py score > $L/xenc_v8_score2_$C.log 2>&1
  echo "== $C CE1 only"; CE_TAGS=    $PY xenc.py blend 2>&1 | grep -v -i warn
  echo "== $C CE1+CE2";  CE_TAGS=",2" $PY xenc.py blend 2>&1 | grep -v -i warn
done
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$BER_ROOT/output_v8 \
  $PY assemble.py $D/fr_p3ce.npz $D/us_p2ce.npz $D/in_p2ce.npz
cd $BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource
$PY utils/validate_submission.py --matching $BER_ROOT/output_v8/matching_results.tsv \
  --candidate $BER_ROOT/output_v8/candidate_pairs.tsv --test-dir dataset/test
echo V8_DONE
