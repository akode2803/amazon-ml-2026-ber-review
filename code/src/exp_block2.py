"""
Candidate-generation sweep on train.

Each (country, source) index is built once; for each scoring exponent `ac` we
retrieve at the largest K and read the smaller Ks off as prefixes (the top-10
is by construction the first 10 of the top-40 under the same score).

Reports the two quantities the competition grades candidate generation on:
candidate recall and candidates per S1 entity.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"


def main():
    split = os.environ.get("SPLIT", "train")
    g = np.load(f"{CACHE}/gt_{split}.npz")
    s1row, pq, ps, tgt_row = g["s1row"], g["pq"], g["ps"], g["tgt_row"]
    n_gt = len(s1row)                      # one GT row per S1 entity
    n_s1rows = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                               columns=["id"]).height
    g2gt = np.full(n_s1rows, -1, np.int64)     # global S1 row -> GT row
    g2gt[s1row] = np.arange(n_gt)
    c1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                         columns=["country"])["country"].to_numpy()
    countries = sorted(set(c1.tolist()))

    ACS = [float(x) for x in os.environ.get("ACS", "0,0.5,1.0").split(",")]
    TOPKS = sorted(int(x) for x in os.environ.get("TOPKS", "10,20,40").split(","))
    KMAX = TOPKS[-1]
    DF_MAX = int(os.environ.get("DF_MAX", "3000"))
    BUDGET = int(os.environ.get("BUDGET", "4000"))
    TAG = os.environ.get("TAG", "v2")

    configs = [(ac, k) for ac in ACS for k in TOPKS]
    acc = {c: dict(found=0, true=0, cand=0,
                   per=np.zeros(n_gt, np.int32),
                   hit=np.zeros(n_gt, np.int32),
                   tot=np.zeros(n_gt, np.int32)) for c in configs}

    for country in countries:
        rows1, ptr1, val1 = block.load_terms(split, 1, country, TAG)
        qpos = np.full(n_s1rows, -1, np.int64)     # global S1 row -> query slot
        qpos[rows1] = np.arange(len(rows1))
        gt_of_slot = g2gt[rows1]                   # query slot -> GT row
        for source in (2, 3):
            t0 = time.time()
            rowsT, ptrT, valT = block.load_terms(split, source, country, TAG)
            idx = block.Index(ptrT, valT, len(rowsT))
            locT = np.full(int(rowsT.max()) + 2, -1, np.int64)
            locT[rowsT] = np.arange(len(rowsT))
            tb = time.time() - t0

            sel = (ps == source) & (qpos[s1row[pq]] >= 0)
            slot = qpos[s1row[pq[sel]]]            # query slot of each true pair
            gtix = pq[sel]                         # GT row of each true pair
            tl = locT[tgt_row[sel]]
            keep = tl >= 0
            slot, gtix, tl = slot[keep], gtix[keep], tl[keep]

            for ac in ACS:
                tq = time.time()
                cand, score, cnt, touch = block.query_index(
                    idx, ptr1, val1, df_max=DF_MAX, budget=BUDGET, topk=KMAX,
                    aq=1.0, ac=ac)
                qt = time.time() - tq
                # rank of each gold target inside its query's ranked list
                rank = np.full(len(slot), KMAX + 1, np.int32)
                for a in range(KMAX):
                    m = (cand[slot, a] == tl) & (rank > KMAX)
                    rank[m] = a
                for k in TOPKS:
                    hit = rank < k
                    nk = np.minimum(cnt, k)
                    a = acc[(ac, k)]
                    a["found"] += int(hit.sum()); a["true"] += len(hit)
                    a["cand"] += int(nk.sum())
                    a["per"][gt_of_slot] += nk
                    np.add.at(a["hit"], gtix, hit.astype(np.int32))
                    np.add.at(a["tot"], gtix, 1)
                print(f"  {country:7s} S{source} ac={ac:<4} "
                      + "  ".join(f"r@{k}={100*(rank<k).mean():5.2f}%" for k in TOPKS)
                      + f"  touched {touch.mean():6.0f}  {qt:4.1f}s", flush=True)
                del cand, score, cnt, touch, rank
            print(f"    [{country} S{source}] {len(rowsT):,} rec, "
                  f"{len(idx.vocab):,} terms, {idx.nbytes()/1e6:.0f} MB, "
                  f"build {tb:.0f}s", flush=True)
            del idx, rowsT, ptrT, valT, locT
            gc.collect()
        del rows1, ptr1, val1, qpos
        gc.collect()

    print(f"\n{'ac':>5} {'topk':>5} {'pair recall':>12} {'ent-complete':>13} "
          f"{'tot pairs':>14} {'mean/S1':>8} {'med':>5} {'p95':>5} {'p99':>5} {'0-cand':>8}")
    for c in configs:
        a = acc[c]
        ent = (a["hit"] == a["tot"]).mean()
        per = a["per"]
        print(f"{c[0]:>5} {c[1]:>5} {a['found']/a['true']*100:11.3f}% "
              f"{ent*100:12.2f}% {a['cand']:>14,} {per.mean():8.1f} "
              f"{np.median(per):5.0f} {np.percentile(per,95):5.0f} "
              f"{np.percentile(per,99):5.0f} {(per==0).mean()*100:7.2f}%")


if __name__ == "__main__":
    main()
