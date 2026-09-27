#!/bin/bash
# Cross-encoder on ALL kept pairs (p2 >= 0.02), not just the 0.02-0.98 band.
# On the LB the CE transferred ~3x better than on val (v6ce 0.976768), so the
# pairs stage 2 is over-confident about (p2 > 0.98) may be where it helps too.
#   phase 1 (after v8): CE1 all-pairs  -> output_v8all  (compare with v8 on val)
#   phase 2 (after v8b): CE1+CE2 all-pairs -> output_v9
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 HF_HUB_OFFLINE=1 \
       BAND_HI=1.01 BAND_TAG=_all
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
VAL="$BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource"
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
MODELS="s2xP3:fr_p3:US,India s2xUS_P2:us_p2:US s2xIN_P2:in_p2:India"
wait_for() { until grep -q "$1" $L/run_seq2.log 2>/dev/null; do
               ps -ef | grep -v grep | grep -q run_seq2.sh || { echo "seq2 gone before $1"; exit 1; }
               sleep 20; done; }
build() {  # build <outdir> <ce tags; - = CE1>
  for m in $MODELS; do
    IFS=: read T d E <<< "$m"
    echo "== $T all-pairs, CE tags '$2'"
    VAL_TAG=$T DUMPS=$d EVAL_CN=$E CE_TAGS=${2//-/} $PY xenc.py blend > $L/xall_b_$1_$d.log 2>&1
    grep -E "blend coef| (US|India) " $L/xall_b_$1_$d.log
  done
  PARAMS=$PARAMS OUTDIR=$BER_ROOT/$1 $PY assemble.py $D/fr_p3ce_all.npz $D/us_p2ce_all.npz \
    $D/in_p2ce_all.npz > $L/assemble_$1.log 2>&1
  grep "decision:" $L/assemble_$1.log
  ( cd $VAL && $PY utils/validate_submission.py --matching $BER_ROOT/$1/matching_results.tsv \
      --candidate $BER_ROOT/$1/candidate_pairs.tsv --test-dir dataset/test > $L/validate_$1.log 2>&1 )
  tail -n 1 $L/validate_$1.log
}

# phase 1 ------------------------------------------------------------------
wait_for V8_DONE
for m in $MODELS; do
  IFS=: read T d E <<< "$m"
  VAL_TAG=$T DUMPS=$d MODEL_TAG= $PY xenc.py data  > $L/xall_d_$d.log 2>&1
  grep -E "band" $L/xall_d_$d.log
  VAL_TAG=$T DUMPS=$d MODEL_TAG= $PY xenc.py score > $L/xall_s1_$d.log 2>&1
done
build output_v8all -
echo V8ALL_DONE

# phase 2 ------------------------------------------------------------------
wait_for V8B_DONE
for m in $MODELS; do
  IFS=: read T d E <<< "$m"
  VAL_TAG=$T DUMPS=$d MODEL_TAG=2 $PY xenc.py score > $L/xall_s2_$d.log 2>&1
done
build output_v9 -,2
echo V9_DONE
