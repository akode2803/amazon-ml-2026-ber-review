#!/bin/bash
# v6 = v5b + self-trained US and India experts: each expert's stage 1 also
# trains on its own country's test pairs, pseudo-labelled from the v4 dumps
# ("explained" negatives), so it sees the test set's extra decoy records.
set -e
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2 CTAG=c2 S2X=1 LGB_THREADS=20 PSEUDO_NEG=explained
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
D=$BER_ROOT/work/dumps
L=$BER_ROOT/logs
while ps -ef | grep -v grep | grep -q "run_night.sh"; do sleep 15; done   # one job at a time
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src
for spec in "US:us:s2xUS_P" "India:in:s2xIN_P"; do
  IFS=: read C c T <<< "$spec"
  TAG=$T COUNTRIES=$C PSEUDO="$C|test|$D/${c}_x.npz|0.98|0.02" $PY stack.py > $L/stack_$T.log 2>&1
  WF_COUNTRIES=$C $PY waterfall.py $T p2 | head -1
  TAG=$T COUNTRIES=$C DUMP=$D/${c}_p.npz $PY infer2.py > $L/infer_$T.log 2>&1
done
PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$BER_ROOT/output_v6 \
  $PY assemble.py $D/fr_p2.npz $D/us_p.npz $D/in_p.npz
cd $BER_ROOT/../6ab10eb3b23ba_student_resource/student_resource
$PY utils/validate_submission.py --matching $BER_ROOT/output_v6/matching_results.tsv \
  --candidate $BER_ROOT/output_v6/candidate_pairs.tsv --test-dir dataset/test
echo V6_DONE
