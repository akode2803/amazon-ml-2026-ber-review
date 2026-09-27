#!/bin/bash
# Phase 2, strictly after v8all (two GPU jobs at once used 22 GB RAM at 06:40):
#   CE2 = cross-encoder trained on train pairs + France pseudo pairs (from fr_p3)
#   v8b  = v7 + mean(CE1, CE2) on the 0.02-0.98 band
#   v9   = v7 + mean(CE1, CE2) on ALL kept pairs
#   v9b  = v9 with the CE coefficient boosted 1.5x (LB trusts the CE more than val)
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 HF_HUB_OFFLINE=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
VAL="$BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource"
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0"
MODELS="s2xP3:fr_p3:US,India s2xUS_P2:us_p2:US s2xIN_P2:in_p2:India"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
until grep -q "V8ALL_DONE" $L/run_all.log 2>/dev/null; do
  ps -ef | grep -v grep | grep -q "run_all.sh" || { echo "run_all ended without V8ALL_DONE"; exit 1; }
  sleep 20
done
while ps -ef | grep -v grep | grep -q "xenc.py"; do sleep 10; done

out() {  # out <outdir> <suffix of the three blended dumps>
  PARAMS=$PARAMS OUTDIR=$BER_ROOT/$1 $PY assemble.py $D/fr_p3ce$2.npz $D/us_p2ce$2.npz \
    $D/in_p2ce$2.npz > $L/assemble_$1.log 2>&1
  grep "decision:" $L/assemble_$1.log
  ( cd $VAL && $PY utils/validate_submission.py --matching $BER_ROOT/$1/matching_results.tsv \
      --candidate $BER_ROOT/$1/candidate_pairs.tsv --test-dir dataset/test > $L/validate_$1.log 2>&1 )
  tail -n 1 $L/validate_$1.log; echo "${1}_DONE"
}
blend() {  # blend <band tag> <out tag> <boost>
  for m in $MODELS; do
    IFS=: read T d E <<< "$m"
    echo "== $T band'$1' out'$2' boost $3"
    VAL_TAG=$T DUMPS=$d EVAL_CN=$E CE_TAGS=",2" BAND_TAG=$1 BAND_HI=$([ -n "$1" ] && echo 1.01 || echo 0.98) \
      OUT_TAG=$2 CE_BOOST=$3 $PY xenc.py blend > $L/p2_b_${d}$2.log 2>&1
    grep -E "blend coef| (US|India) " $L/p2_b_${d}$2.log
  done
}

# CE2 ----------------------------------------------------------------------
VAL_TAG=s2xP3 DUMPS=fr_p3 MODEL_TAG=2 PSEUDO_DUMP=France:fr_p3:600000 $PY xenc.py data > $L/xenc2_data.log 2>&1
MODEL_TAG=2 $PY xenc.py train > $L/xenc2_train.log 2>&1
grep -E "holdout" $L/xenc2_train.log

# v8b: CE2 on the default band (bands + CE1 scores exist from v8) -----------
for m in $MODELS; do
  IFS=: read T d E <<< "$m"
  VAL_TAG=$T DUMPS=$d MODEL_TAG=2 $PY xenc.py score > $L/p2_s2_$d.log 2>&1
done
blend "" "" 1
out output_v8b ""

# v9 / v9b: CE2 on all pairs (all-pair bands + CE1 scores exist from v8all) --
for m in $MODELS; do
  IFS=: read T d E <<< "$m"
  VAL_TAG=$T DUMPS=$d MODEL_TAG=2 BAND_TAG=_all BAND_HI=1.01 $PY xenc.py score > $L/p2_s2all_$d.log 2>&1
done
blend _all _all 1
out output_v9 _all
blend _all _allb 1.5
out output_v9b _allb
