#!/bin/bash
# v14 = v13 + transitive cross-encoder: stage 3 also sees the large CE's score of
# (candidate record, entity's confident anchor record).  Then the India dense
# pairs are merged again on top (K = 2, 3).
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
B=/c/Users/LENOVO/Desktop/Amazon_ML/ber2
L=$B/logs
KEY="-i $HOME/.ssh/runpod_ed25519 -o IdentitiesOnly=yes -o ConnectTimeout=20"
S="ssh $KEY -p <POD_PORT> root@<POD_IP>"
VAL="$B/../6ab10eb3b23ba_student_resource/student_resource"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src

until grep -qE "V13_DONE" $L/run_v13.log 2>/dev/null; do
  ps -ef | grep -v grep | grep -q run_v13.sh || { echo "v13 ended without V13_DONE"; exit 1; }
  sleep 30
done
until $S 'grep -qE "score done|Traceback" /workspace/ce/score_anchor.log 2>/dev/null'; do sleep 30; done
$S 'grep -q "score done" /workspace/ce/score_anchor.log'
scp -q $KEY -P <POD_PORT> root@<POD_IP>:/workspace/ce/logits_anchor.npy $B/work/runpod/
echo ANCHOR_SCORED

for m in s2xP3:fr_p3:US,India s2xUS_P2:us_p2:US s2xIN_P2:in_p2:India; do
  IFS=: read T d E <<< "$m"
  S3_ANCHOR=1 VAL_TAG=$T DUMPS=$d EVAL_CN=$E CE_TAGS=",2,3,4,5" OUT_TAG=_c5a $PY stage3.py 2>&1 | grep -v -i warn
done
echo "reference stage3 _c5: US 0.98952 | India 0.98178 | France-model US 0.98949 India 0.98181"

for K in 2 3; do
  echo "=== K=$K"
  VAL_TAG=s2xIN_P2s3_c5a DUMP=in_p2s3_c5a DK=$K $PY dense_merge.py 2>&1 | grep -v -i warn
  od=output_v14_k$K
  PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$B/$od $PY assemble.py \
    $B/work/dumps/fr_p3s3_c5a.npz $B/work/dumps/us_p2s3_c5a.npz $B/work/dumps/in_p2s3_c5ad$K.npz \
    > $L/assemble_$od.log 2>&1
  grep -E "decision:" $L/assemble_$od.log
  ( cd $VAL && $PY utils/validate_submission.py --matching $B/$od/matching_results.tsv \
      --candidate $B/$od/candidate_pairs.tsv --test-dir dataset/test 2>&1 | tail -n 1 )
done
echo V14_DONE
