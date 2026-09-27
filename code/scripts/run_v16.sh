#!/bin/bash
# v16 = v13_k3 + the retrieval fix for US and France (retriever continued from the
# India-tuned model on India + US + France-pseudo pairs; FAISS IVF search).
#   output_v16    India K3 + US K3 + France K3
#   output_v16us  India K3 + US K3, France unchanged (safer comparison)
set -eo pipefail
export PYTHONIOENCODING=utf-8 BER_ROOT=C:/Users/LENOVO/Desktop/Amazon_ML/ber2
PY=/c/Users/LENOVO/anaconda3/envs/aml/python.exe
B=/c/Users/LENOVO/Desktop/Amazon_ML/ber2
R=$B/work/runpod
D=$B/work/dumps
L=$B/logs
KEY="-i $HOME/.ssh/runpod_ed25519 -o IdentitiesOnly=yes -o ConnectTimeout=20"
H=root@<POD_IP>
S="ssh $KEY -p <POD_PORT> $H"
VAL="$B/../6ab10eb3b23ba_student_resource/student_resource"
cd /c/Users/LENOVO/Desktop/Amazon_ML/ber/src

until $S 'grep -qE "BIENC_DONE|Traceback" /workspace/ce/bienc2.log'; do sleep 30; done
$S 'grep -q BIENC_DONE /workspace/ce/bienc2.log'
for f in train_s2_US train_s3_US test_s2_US test_s3_US test_s2_FR test_s3_FR; do
  scp -q $KEY -P <POD_PORT> $H:/workspace/ce/dense2_knn_$f.npz $R/
done
DK=5 COUNTRY=US SUF=_US $PY dense_newpairs.py 2>&1 | grep -v -i warn
DK=5 COUNTRY=France SUF=_FR SPLITS=test $PY dense_newpairs.py 2>&1 | grep -v -i warn
scp -q $KEY -P <POD_PORT> $R/dense_new_US.parquet $R/dense_new_FR.parquet $H:/workspace/ce/
$S 'cd /workspace/ce && rm -f score_new2.log && (nohup sh -c "WORKERS=8 SCORE_IN=dense_new_US.parquet SCORE_OUT=logits_dense_new_US.npy python remote_ce.py score && WORKERS=8 SCORE_IN=dense_new_FR.parquet SCORE_OUT=logits_dense_new_FR.npy python remote_ce.py score && echo NEW2_DONE" > score_new2.log 2>&1 &)'
until $S 'grep -qE "NEW2_DONE|Traceback" /workspace/ce/score_new2.log'; do sleep 30; done
$S 'grep -q NEW2_DONE /workspace/ce/score_new2.log'
scp -q $KEY -P <POD_PORT> $H:/workspace/ce/logits_dense_new_US.npy $H:/workspace/ce/logits_dense_new_FR.npy $R/
echo NEW2_SCORED

for K in 2 3; do
  echo "=== US K=$K"
  FIT=_US:s2xUS_P2s3_c5 APPLY_SUF=_US DUMP=us_p2s3_c5 DK=$K $PY dense_merge2.py 2>&1 | grep -v -i warn
  echo "=== France K=$K (model fitted on US + India val)"
  FIT=:s2xIN_P2s3_c5,_US:s2xUS_P2s3_c5 APPLY_SUF=_FR DUMP=fr_p3s3_c5 DK=$K $PY dense_merge2.py 2>&1 | grep -v -i warn
done

build() {  # build <outdir> <fr dump> <us dump> <in dump>
  PARAMS="US:1.1:1.0,India:1.1:1.0,France:1.1:4.0" OUTDIR=$B/$1 $PY assemble.py $D/$2.npz $D/$3.npz $D/$4.npz \
    > $L/assemble_$1.log 2>&1
  grep -E "t=1.1|decision:" $L/assemble_$1.log
  ( cd $VAL && $PY utils/validate_submission.py --matching $B/$1/matching_results.tsv \
      --candidate $B/$1/candidate_pairs.tsv --test-dir dataset/test 2>&1 | tail -n 1 )
  echo "${1}_DONE"
}
build output_v16   fr_p3s3_c5d3 us_p2s3_c5d3 in_p2s3_c5d3
build output_v16us fr_p3s3_c5   us_p2s3_c5d3 in_p2s3_c5d3
echo V16_DONE
