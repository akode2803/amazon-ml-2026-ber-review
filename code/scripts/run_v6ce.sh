#!/bin/bash
# v6ce = v6 + cross-encoder blend on every country's uncertain band.  Each
# blend is fitted on the val file of the model whose p2 it re-scores
# (France: joint s2xP2, already written as fr_p2ce; US/India: v6 experts).
set -e
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 HF_HUB_OFFLINE=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
until grep -qE "V6_DONE" $L/run_v6.log 2>/dev/null; do
  ps -ef | grep -v grep | grep -q "run_v6.sh" || { echo "v6 ended without V6_DONE"; exit 1; }
  sleep 20
done
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
for spec in "US:s2xUS_P:us_p" "India:s2xIN_P:in_p"; do
  IFS=: read C T d <<< "$spec"
  export VAL_TAG=$T DUMPS=$d EVAL_CN=$C
  $PY xenc.py data  > $L/xenc_data_$C.log 2>&1
  $PY xenc.py score > $L/xenc_score_$C.log 2>&1
  $PY xenc.py blend 2>&1 | grep -v -i warn | tee $L/xenc_blend_$C.log
done
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$BER_ROOT/output_v6ce \
  $PY assemble.py $D/fr_p2ce.npz $D/us_pce.npz $D/in_pce.npz
cd $BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource
$PY utils/validate_submission.py --matching $BER_ROOT/output_v6ce/matching_results.tsv \
  --candidate $BER_ROOT/output_v6ce/candidate_pairs.tsv --test-dir dataset/test
echo V6CE_DONE
