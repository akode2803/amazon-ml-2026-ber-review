"""
Why does retrieval miss true pairs that share plenty of tokens?

For every val-fold true pair missed by the production candidate file (both
directions), re-run the *forward* query of its S1 record under relaxed
settings and see which relaxation first retrieves it:

    rank      production budget/df_max, top-200 instead of top-20
    budget    budget 40k, df_max 3000, top-20
    dfmax     budget 40k, df_max 30k, top-20
    both      budget 40k, df_max 30k, top-200

plus the number of shared blocking terms and the smallest df among them.
Usage: python exp_miss.py [ctag] [country] [source]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block  # noqa: E402
import model as M  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"


def rank_of(cand, cnt, target):
    k = cand.shape[1]
    ok = (np.arange(k)[None, :] < cnt[:, None]) & (cand == target[:, None])
    return np.where(ok.any(1), ok.argmax(1), -1)


def main():
    ctag = sys.argv[1] if len(sys.argv) > 1 else "c1"
    country = sys.argv[2] if len(sys.argv) > 2 else "India"
    source = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    split = "train"
    s1row, pq, ps, tgt_row = M.gt_arrays(split)
    ids1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                           columns=["id"])["id"].to_numpy()
    fold = M.fold_of(ids1)
    z = np.load(f"{CACHE}/{split}/cand/{ctag}_{country}_s{source}.npz")
    rows1c, rowsTc, q, c = z["rows1"], z["rowsT"], z["q"], z["c"]
    rows1, ptr1, val1 = block.load_terms(split, 1, country)
    rowsT, ptrT, valT = block.load_terms(split, source, country)
    assert (rows1 == rows1c).all() and (rowsT == rowsTc).all()
    n1, nT = len(rows1), len(rowsT)
    qpos = np.full(len(ids1) + 2, -1, np.int64); qpos[rows1] = np.arange(n1)
    locT = np.full(int(rowsT.max()) + 2, -1, np.int64); locT[rowsT] = np.arange(nT)
    sel = ps == source
    gq = qpos[s1row[pq[sel]]]
    gc = locT[np.minimum(tgt_row[sel], len(locT) - 1)]
    ok = (gq >= 0) & (gc >= 0)
    gq, gc = gq[ok], gc[ok]
    ok = fold[rows1[gq]] == 0
    gq, gc = gq[ok], gc[ok]
    have = np.sort(q.astype(np.int64) * nT + c)
    key = gq * nT + gc
    j = np.clip(np.searchsorted(have, key), 0, len(have) - 1)
    miss = have[j] != key
    mq, mc = gq[miss], gc[miss]
    print(f"{country} S{source}: val true pairs {len(key):,}, missed {miss.sum():,} "
          f"({miss.mean() * 100:.2f}%)")

    idx = block.Index(ptrT, valT, nT)
    lens = ptr1[mq + 1] - ptr1[mq]
    qp = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
    qv = np.concatenate([val1[ptr1[a]:ptr1[a + 1]] for a in mq])

    nshared = np.zeros(len(mq), np.int32)
    mindf = np.full(len(mq), 1 << 30, np.int64)
    spent_before = np.zeros(len(mq), np.int64)
    tq = block.map_terms(idx, qv)
    for i in range(len(mq)):
        a = set(valT[ptrT[mc[i]]:ptrT[mc[i] + 1]].tolist())
        tt = tq[qp[i]:qp[i + 1]]
        vv = qv[qp[i]:qp[i + 1]]
        dfq = np.where(tt >= 0, idx.df[np.maximum(tt, 0)], 1 << 30)
        sh = np.array([v in a for v in vv.tolist()])
        nshared[i] = sh.sum()
        if sh.any():
            mindf[i] = dfq[sh].min()
            o = np.argsort(dfq, kind="stable")
            first = np.nonzero(sh[o])[0][0]
            w = dfq[o][:first]
            spent_before[i] = w[w < (1 << 30)].sum()
    print(f"  shared terms: none {np.mean(nshared == 0) * 100:.1f}%  "
          f"median {np.median(nshared):.0f}")
    print(f"  min df of shared: <=100 {np.mean(mindf <= 100) * 100:.1f}%  "
          f"<=3000 {np.mean(mindf <= 3000) * 100:.1f}%  "
          f">3000 {np.mean((mindf > 3000) & (nshared > 0)) * 100:.1f}%")
    print(f"  budget spent before first shared term: >4000 "
          f"{np.mean(spent_before > 4000) * 100:.1f}%  median {np.median(spent_before):.0f}")

    target = mc.astype(np.int32)
    res = {}
    for name, bud, dfm, k in (("prod(sanity)", 4000, 3000, 20),
                              ("rank top200", 4000, 3000, 200),
                              ("budget 40k", 40000, 3000, 20),
                              ("budget 40k dfmax 30k", 40000, 30000, 20),
                              ("both, top200", 40000, 30000, 200)):
        t0 = time.time()
        cand, sc, cnt, _ = block.query_index(idx, qp, qv, df_max=dfm, budget=bud,
                                             topk=k, max_touch=200000)
        r = rank_of(cand, cnt, target)
        res[name] = r
        h = r >= 0
        print(f"  {name:<24} recovered {h.mean() * 100:5.1f}%  "
              f"rank p50 {np.median(r[h]) if h.any() else -1:.0f}  "
              f"{time.time() - t0:.1f}s", flush=True)
    r = res["rank top200"]
    for lo, hi in ((20, 30), (30, 50), (50, 100), (100, 200)):
        print(f"    rank in [{lo},{hi}): {np.mean((r >= lo) & (r < hi)) * 100:5.1f}%")


if __name__ == "__main__" and sys.argv[1:2] != ["rev"]:
    main()


def reverse_ranks():
    """Same missed pairs, reverse direction: rank of the true S1 when the
    target record queries the S1 index."""
    ctag = sys.argv[2] if len(sys.argv) > 2 else "c1"
    country = sys.argv[3] if len(sys.argv) > 3 else "India"
    source = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    split = "train"
    s1row, pq, ps, tgt_row = M.gt_arrays(split)
    ids1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                           columns=["id"])["id"].to_numpy()
    fold = M.fold_of(ids1)
    z = np.load(f"{CACHE}/{split}/cand/{ctag}_{country}_s{source}.npz")
    q, c = z["q"], z["c"]
    rows1, ptr1, val1 = block.load_terms(split, 1, country)
    rowsT, ptrT, valT = block.load_terms(split, source, country)
    n1, nT = len(rows1), len(rowsT)
    qpos = np.full(len(ids1) + 2, -1, np.int64); qpos[rows1] = np.arange(n1)
    locT = np.full(int(rowsT.max()) + 2, -1, np.int64); locT[rowsT] = np.arange(nT)
    sel = ps == source
    gq = qpos[s1row[pq[sel]]]
    gc = locT[np.minimum(tgt_row[sel], len(locT) - 1)]
    ok = (gq >= 0) & (gc >= 0)
    gq, gc = gq[ok], gc[ok]
    ok = fold[rows1[gq]] == 0
    gq, gc = gq[ok], gc[ok]
    have = np.sort(q.astype(np.int64) * nT + c)
    key = gq * nT + gc
    j = np.clip(np.searchsorted(have, key), 0, len(have) - 1)
    miss = have[j] != key
    mq, mc = gq[miss], gc[miss]
    idx1 = block.Index(ptr1, val1, n1)
    lens = ptrT[mc + 1] - ptrT[mc]
    qp = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
    qv = np.concatenate([valT[ptrT[a]:ptrT[a + 1]] for a in mc])
    for name, bud, dfm, k in (("rev prod top200", 4000, 3000, 200),
                              ("rev both top200", 40000, 30000, 200)):
        cand, sc, cnt, _ = block.query_index(idx1, qp, qv, df_max=dfm, budget=bud,
                                             topk=k, max_touch=200000)
        r = rank_of(cand, cnt, mq.astype(np.int32))
        h = r >= 0
        print(f"  {name:<20} missed {len(mq):,} recovered {h.mean() * 100:5.1f}%  "
              + "  ".join(f"<{b}:{np.mean(h & (r < b)) * 100:.1f}%"
                          for b in (10, 20, 50, 100)), flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["rev"]:
    reverse_ranks()
