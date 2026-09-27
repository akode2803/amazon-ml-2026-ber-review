#!/bin/bash
# v7 = round-2 self-training everywhere (India proxy: round 1 0.90427 -> round 2
# 0.91223).  Each model is retrained on test pairs pseudo-labelled by its own
# round-1 output: France from v5b (fr_p2), US/India from v6 (us_p, in_p).
# Then a round-3 proxy on India to see whether a third round still pays.
set -e
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 LGB_THREADS=20 PSEUDO_NEG=explained
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
while ps -ef | grep -v grep | grep -q "run_v6.sh"; do sleep 15; done   # one job at a time
test -f $D/us_p.npz && test -f $D/in_p.npz
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
TAG=s2xP3 PSEUDO="France|test|$D/fr_p2.npz|0.98|0.02" $PY stack.py > $L/stack_s2xP3.log 2>&1
TAG=s2xP3 COUNTRIES=France DUMP=$D/fr_p3.npz $PY infer2.py > $L/infer_s2xP3.log 2>&1
for spec in "US:us:s2xUS_P2" "India:in:s2xIN_P2"; do
  IFS=: read C c T <<< "$spec"
  TAG=$T COUNTRIES=$C PSEUDO="$C|test|$D/${c}_p.npz|0.98|0.02" $PY stack.py > $L/stack_$T.log 2>&1
  WF_COUNTRIES=$C $PY waterfall.py $T p2 | head -1
  TAG=$T COUNTRIES=$C DUMP=$D/${c}_p2.npz $PY infer2.py > $L/infer_$T.log 2>&1
done
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$BER_ROOT/output_v7 \
  $PY assemble.py $D/fr_p3.npz $D/us_p2.npz $D/in_p2.npz
( cd $BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource && \
  $PY utils/validate_submission.py --matching $BER_ROOT/output_v7/matching_results.tsv \
    --candidate $BER_ROOT/output_v7/candidate_pairs.tsv --test-dir dataset/test )
echo V7_DONE

# round-3 proxy: pseudo-label India again with the round-2 US-only model
SPLIT=train TAG=locoUS_ps3 COUNTRIES=India DUMP=$D/ps3_IN.npz $PY infer2.py > $L/infer_ps3_IN.log 2>&1
TAG=locoUS_ps4 COUNTRIES=US EVAL_COUNTRIES=India \
  PSEUDO="India|train|$D/ps3_IN.npz|0.98|0.02" $PY stack.py > $L/stack_locoUS_ps4.log 2>&1
WF_COUNTRIES=India $PY waterfall.py locoUS_ps4 p2 > $L/wf_locoUS_ps4_India.log 2>&1
echo "RESULT $(head -1 $L/wf_locoUS_ps4_India.log)"
echo R3_DONE
