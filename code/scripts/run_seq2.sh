#!/bin/bash
# Strictly sequential (parallel jobs exhausted RAM at 03:00):
#   1 v6ce  = v6 + CE1 blend (US blend already done -> us_pce.npz)
#   2 v7    = round-2 self-training everywhere (France stage 1 already saved)
#   3 v8    = v7 + CE1 blend
#   4 v8b   = v7 + mean(CE1, CE2), CE2 = CE trained with France pseudo pairs
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 \
       LGB_THREADS=20 PSEUDO_NEG=explained HF_HUB_OFFLINE=1
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
VAL="$BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource"
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
validate() {
  ( cd $VAL && $PY utils/validate_submission.py --matching $BER_ROOT/$1/matching_results.tsv \
      --candidate $BER_ROOT/$1/candidate_pairs.tsv --test-dir dataset/test > $L/validate_$1.log 2>&1 )
  tail -n 1 $L/validate_$1.log
}
assemble() {  # assemble <outdir> <dumps...>
  local o=$1; shift
  PARAMS=$PARAMS OUTDIR=$BER_ROOT/$o $PY assemble.py "$@" > $L/assemble_$o.log 2>&1
  grep "decision:" $L/assemble_$o.log
}
wf() { WF_COUNTRIES=$2 $PY waterfall.py $1 p2 > $L/wf_$1.log 2>&1; head -n 1 $L/wf_$1.log; }
ce() {  # ce <val_tag> <dump> <eval countries> <ce tags; "-" = CE1>
  VAL_TAG=$1 DUMPS=$2 EVAL_CN=$3 MODEL_TAG= $PY xenc.py data > $L/xenc_d_$2.log 2>&1
  for t in ${4//,/ }; do
    VAL_TAG=$1 DUMPS=$2 MODEL_TAG=${t/-/} $PY xenc.py score > $L/xenc_s${t}_$2.log 2>&1
  done
  VAL_TAG=$1 DUMPS=$2 EVAL_CN=$3 CE_TAGS=${4//-/} $PY xenc.py blend > $L/xenc_b_$2.log 2>&1
  grep -E "blend coef| (US|India) " $L/xenc_b_$2.log
}

# 1 --------------------------------------------------------------- v6ce
ce s2xIN_P in_p India -
assemble output_v6ce $D/fr_p2ce.npz $D/us_pce.npz $D/in_pce.npz
validate output_v6ce; echo V6CE_DONE

# 2 ----------------------------------------------------------------- v7
TAG=s2xP3 PSEUDO="France|test|$D/fr_p2.npz|0.98|0.02" $PY stack.py > $L/stack_s2xP3.log 2>&1
TAG=s2xP3 COUNTRIES=France DUMP=$D/fr_p3.npz $PY infer2.py > $L/infer_s2xP3.log 2>&1
for spec in "US:us:s2xUS_P2" "India:in:s2xIN_P2"; do
  IFS=: read C c T <<< "$spec"
  TAG=$T COUNTRIES=$C PSEUDO="$C|test|$D/${c}_p.npz|0.98|0.02" $PY stack.py > $L/stack_$T.log 2>&1
  wf $T $C
  TAG=$T COUNTRIES=$C DUMP=$D/${c}_p2.npz $PY infer2.py > $L/infer_$T.log 2>&1
done
assemble output_v7 $D/fr_p3.npz $D/us_p2.npz $D/in_p2.npz
validate output_v7; echo V7_DONE

# 3 ----------------------------------------------------------------- v8
ce s2xP3 fr_p3 US,India -
ce s2xUS_P2 us_p2 US -
ce s2xIN_P2 in_p2 India -
assemble output_v8 $D/fr_p3ce.npz $D/us_p2ce.npz $D/in_p2ce.npz
validate output_v8; echo V8_DONE

# 4 ---------------------------------------------------------------- v8b
VAL_TAG=s2xP3 DUMPS=fr_p3 MODEL_TAG=2 PSEUDO_DUMP=France:fr_p3:600000 $PY xenc.py data > $L/xenc2_data.log 2>&1
MODEL_TAG=2 $PY xenc.py train > $L/xenc2_train.log 2>&1
grep -E "holdout" $L/xenc2_train.log
ce s2xP3 fr_p3 US,India -,2
ce s2xUS_P2 us_p2 US -,2
ce s2xIN_P2 in_p2 India -,2
assemble output_v8b $D/fr_p3ce.npz $D/us_p2ce.npz $D/in_p2ce.npz
validate output_v8b; echo V8B_DONE
