#!/bin/bash
# v13 = v12s3 + India pairs recovered by the fine-tuned multilingual retriever
# (FAISS IVF top-K, not in the lexical candidate set), scored by the large CE,
# merged via dense_merge.py.  Built for K = 2, 3, 5 -> choose on val.
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
B=/c/Users/LENOVO/Desktop/Amazon_ML/ber2
R=$B/work/runpod
L=$B/logs
KEY="-i $HOME/.ssh/runpod_ed25519 -o IdentitiesOnly=yes -o ConnectTimeout=20"
S="ssh $KEY -p <POD_PORT> root@<POD_IP>"
VAL="$B/../6ab10eb3b23ba_student_resource/student_resource"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src

until $S 'grep -qE "BIENC_DONE|Traceback" /workspace/ce/bienc.log'; do sleep 30; done
$S 'grep -q BIENC_DONE /workspace/ce/bienc.log'
scp -q $KEY -P <POD_PORT> root@<POD_IP>:/workspace/ce/dense2_knn_test_s2.npz \
    root@<POD_IP>:/workspace/ce/dense2_knn_test_s3.npz $R/
DK=5 $PY dense_newpairs.py 2>&1 | grep -v -i warn
scp -q $KEY -P <POD_PORT> $R/dense_new.parquet root@<POD_IP>:/workspace/ce/
$S 'cd /workspace/ce && rm -f score_dense.log && (WORKERS=8 SCORE_IN=dense_new.parquet SCORE_OUT=logits_dense_new.npy nohup python remote_ce.py score > score_dense.log 2>&1 &)'
until $S 'grep -qE "score done|Traceback" /workspace/ce/score_dense.log'; do sleep 30; done
$S 'tail -n 1 /workspace/ce/score_dense.log; grep -q "score done" /workspace/ce/score_dense.log'
scp -q $KEY -P <POD_PORT> root@<POD_IP>:/workspace/ce/logits_dense_new.npy $R/
echo DENSE_SCORED

for K in 2 3 5; do
  echo "=== K=$K"
  DK=$K $PY dense_merge.py 2>&1 | grep -v -i warn
  od=output_v13_k$K
  PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$B/$od $PY assemble.py \
    $B/work/dumps/fr_p3s3_c5.npz $B/work/dumps/us_p2s3_c5.npz $B/work/dumps/in_p2s3_c5d$K.npz \
    > $L/assemble_$od.log 2>&1
  grep -E "India|decision:" $L/assemble_$od.log | grep -v " S1 "
  ( cd $VAL && $PY utils/validate_submission.py --matching $B/$od/matching_results.tsv \
      --candidate $B/$od/candidate_pairs.tsv --test-dir dataset/test 2>&1 | tail -n 1 )
done
echo V13_DONE
