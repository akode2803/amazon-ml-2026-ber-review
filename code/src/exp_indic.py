import os
"""
How much does Indic romanization + the phonetic skeleton actually buy?

Isolates the true pairs whose S2/S3 record is written in an Indic script (the
S1 reference side is always Latin) and measures candidate recall on that slice,
with and without the skeleton term namespaces.  These are the pairs where the
two names share literally zero characters, so if the skeleton does not carry
them, nothing in the name channel can.
"""
import os, re, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
DATA = f"{ROOT}/student_resource/dataset"
INDIC = re.compile(r"[\u0900-\u0D7F]")

g = np.load(f"{CACHE}/gt_train.npz")
s1row, pq, ps, tgt_row = g["s1row"], g["pq"], g["ps"], g["tgt_row"]
n_s1rows = pl.read_parquet(f"{CACHE}/train/s1/meta.parquet", columns=["id"]).height

tot_all = hit_all = 0
tot_ind = hit_ind = 0
tot_ind_noskel = 0
for source in (2, 3):
    raw = pl.read_csv(f"{DATA}/train/train_source{source}.tsv", separator="\t",
                      quote_char=None, columns=["business_name"])["business_name"]
    is_indic = np.array([bool(INDIC.search(x)) if x else False for x in raw.to_list()])
    del raw
    z = np.load(f"{CACHE}/train/cand/c1_India_s{source}.npz")
    rows1, rowsT, q, c = z["rows1"], z["rowsT"], z["q"], z["c"]
    nT = len(rowsT)
    qpos = np.full(n_s1rows, -1, np.int64); qpos[rows1] = np.arange(len(rows1))
    locT = np.full(int(rowsT.max()) + 2, -1, np.int64); locT[rowsT] = np.arange(nT)
    sel = (ps == source) & (qpos[s1row[pq]] >= 0)
    tl = locT[tgt_row[sel]]
    ok = tl >= 0
    gkey = qpos[s1row[pq[sel]]][ok] * nT + tl[ok]
    gidc = is_indic[tgt_row[sel]][ok]
    key = np.sort(q.astype(np.int64) * nT + c)
    j = np.searchsorted(key, gkey); j = np.clip(j, 0, len(key) - 1)
    hit = key[j] == gkey
    tot_all += len(hit); hit_all += int(hit.sum())
    tot_ind += int(gidc.sum()); hit_ind += int(hit[gidc].sum())
    # would a NON-skeleton term have found it?  re-derive shared terms without
    # the k:/b:/s:/2:/3:/4:/5: namespaces by checking latin-token overlap only
    m1 = pl.read_parquet(f"{CACHE}/train/s1/meta.parquet",
                         columns=["name_toks", "addr_toks", "nums"])
    mt = pl.read_parquet(f"{CACHE}/train/s{source}/meta.parquet",
                         columns=["name_toks", "addr_toks", "nums"])
    gq_glob = s1row[pq[sel]][ok][gidc]
    gt_glob = tgt_row[sel][ok][gidc]
    a_nt = m1["name_toks"].to_numpy(); b_nt = mt["name_toks"].to_numpy()
    share = 0
    for x, y in zip(a_nt[gq_glob], b_nt[gt_glob]):
        if x and y and (set(x.split("\t")) & set(y.split("\t"))):
            share += 1
    tot_ind_noskel += share
    print(f"  S{source}: indic-script targets in GT {int(gidc.sum()):,}; "
          f"of those {share:,} ({share/max(int(gidc.sum()),1)*100:.2f}%) still "
          f"share a raw latin name token", flush=True)
    del z, m1, mt

print(f"\nIndia candidate recall, all true pairs      : "
      f"{hit_all/tot_all*100:.3f}%  ({hit_all:,}/{tot_all:,})")
print(f"India candidate recall, Indic-script targets: "
      f"{hit_ind/max(tot_ind,1)*100:.3f}%  ({hit_ind:,}/{tot_ind:,})")
print(f"Indic-script true pairs sharing ANY latin name token: "
      f"{tot_ind_noskel/max(tot_ind,1)*100:.2f}%  -- without romanization the "
      f"name channel is blind on the other {100-tot_ind_noskel/max(tot_ind,1)*100:.2f}%")
