"""
Error taxonomy for candidate generation: for one (country, source) partition,
classify every true pair by *why* it is or isn't retrieved.

Buckets
    A. no shared term at all                -> normalization failure
    B. shared terms exist, all df > DF_MAX   -> only common evidence
    C. shared rare term, but budget spent before reaching it
    D. touched, ranked outside top-K         -> ranking failure
    E. retrieved
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
TYPE = {"n": "name-tok", "k": "name-skel", "a": "addr-tok", "b": "addr-skel",
        "#": "number", "d": "domain5g"}


def main(country="India", source=2, topk=20, df_max=3000, budget=4000,
         nsample=40000):
    split = "train"
    g = np.load(f"{CACHE}/gt_{split}.npz")
    s1row, pq, ps, tgt_row = g["s1row"], g["pq"], g["ps"], g["tgt_row"]
    c1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                         columns=["country"])["country"].to_numpy()

    keep, tptr_t, tval_t, idx = block.build_index(split, source, country)
    n_glob = pl.read_parquet(f"{CACHE}/{split}/s{source}/meta.parquet",
                             columns=["id"]).height
    gl2loc = np.full(n_glob, -1, np.int32)
    gl2loc[keep] = np.arange(len(keep), dtype=np.int32)

    qsel = (c1[s1row] == country).nonzero()[0]
    inq = np.zeros(len(s1row), bool); inq[qsel] = True
    mask = (ps == source) & inq[pq]
    sub_pq, sub_tr = pq[mask], gl2loc[tgt_row[mask]]
    ok = sub_tr >= 0
    sub_pq, sub_tr = sub_pq[ok], sub_tr[ok]
    print(f"{country} S{source}: {len(qsel):,} queries, {len(sub_pq):,} true pairs")

    # ---- query terms (CSR over the GT-ordered query list)
    pos_in_q = np.full(len(s1row), -1, np.int64); pos_in_q[qsel] = np.arange(len(qsel))
    q_global = s1row[qsel].astype(np.int64)
    z = np.load(f"{CACHE}/{split}/s1/terms.npz")
    a_ptr, a_val = z["tptr"], z["tval"]
    lens = (a_ptr[q_global + 1] - a_ptr[q_global]).astype(np.int64)
    q_ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
    q_val = np.empty(q_ptr[-1], np.uint32)
    block._gather_csr(a_ptr, a_val, q_global, q_ptr, q_val)
    del z, a_ptr, a_val
    q_norm = block.query_norms(q_ptr, q_val, idx)

    # ---- run retrieval with a wide top-k so ranking failures are visible
    WIDE = 400
    t0 = time.time()
    cand, score, cnt = block.query_index(idx, q_ptr, q_val, q_norm,
                                         df_max=df_max, budget=budget, topk=WIDE)
    print(f"retrieval @{WIDE}: {time.time()-t0:.0f}s, mean touched-kept {cnt.mean():.1f}")

    # ---- rank of each gold target
    rows = pos_in_q[sub_pq]
    rank = np.full(len(rows), -1, np.int32)
    for a in range(WIDE):
        m = (cand[rows, a] == sub_tr) & (rank < 0)
        rank[m] = a
    print(f"\nrecall@{topk}   {(rank>=0).mean()*0+((rank>=0)&(rank<topk)).mean()*100:6.2f}%")
    for k in (1, 3, 5, 10, 20, 50, 100, 200, 400):
        print(f"  recall@{k:<4d} {((rank >= 0) & (rank < k)).mean()*100:6.2f}%")

    # ---- shared-term analysis on the pairs that never appeared at all
    rng = np.random.default_rng(0)
    missed = (rank < 0).nonzero()[0]
    hitidx = ((rank >= 0) & (rank < topk)).nonzero()[0]
    print(f"\nnever retrieved (@{WIDE}): {len(missed):,} / {len(rank):,} "
          f"= {len(missed)/len(rank)*100:.2f}%")

    def shared_profile(sample, label):
        nsh, mindf, bytype = [], [], {}
        for i in sample:
            qi = rows[i]
            qt = set(q_val[q_ptr[qi]:q_ptr[qi + 1]].tolist())
            tr = sub_tr[i]
            tt = set(tval_t[tptr_t[tr]:tptr_t[tr + 1]].tolist())
            sh = qt & tt
            nsh.append(len(sh))
            if sh:
                pos = np.searchsorted(idx.vocab, np.fromiter(sh, np.uint32, len(sh)))
                pos = np.clip(pos, 0, len(idx.vocab) - 1)
                d = idx.df[pos]
                mindf.append(int(d.min()))
            else:
                mindf.append(1 << 30)
        nsh = np.asarray(nsh); mindf = np.asarray(mindf)
        print(f"\n  [{label}] n={len(sample):,}")
        print(f"    shared terms: 0 -> {(nsh==0).mean()*100:5.2f}%   "
              f"mean {nsh.mean():.2f}  median {np.median(nsh):.0f}")
        print(f"    min df of shared terms:  <=10 {(mindf<=10).mean()*100:5.1f}%  "
              f"<=100 {(mindf<=100).mean()*100:5.1f}%  <=1000 {(mindf<=1000).mean()*100:5.1f}%  "
              f"<=df_max {(mindf<=df_max).mean()*100:5.1f}%  none {(nsh==0).mean()*100:5.1f}%")
        return nsh, mindf

    sm = rng.choice(missed, min(nsample, len(missed)), replace=False) if len(missed) else []
    sh_m, df_m = shared_profile(sm, "NEVER RETRIEVED")
    sh_h, df_h = shared_profile(rng.choice(hitidx, min(nsample, len(hitidx)),
                                           replace=False), "RETRIEVED")

    # ---- bucket the misses
    nA = (sh_m == 0).sum()
    nB = ((sh_m > 0) & (df_m > df_max)).sum()
    nC = ((sh_m > 0) & (df_m <= df_max)).sum()
    tot = len(sh_m)
    print(f"\n  MISS TAXONOMY (of never-retrieved sample, n={tot:,})")
    print(f"    A no shared term at all        {nA/tot*100:5.1f}%")
    print(f"    B shared but all df > {df_max:<6d}  {nB/tot*100:5.1f}%")
    print(f"    C rare term shared, budget/cap {nC/tot*100:5.1f}%")
    nD = ((rank >= topk)).sum()
    print(f"    D touched but rank >= {topk}      {nD:,} pairs "
          f"({nD/len(rank)*100:.2f}% of all true pairs)")

    # ---- show concrete examples of category A and C
    m1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet")
    m2 = pl.read_parquet(f"{CACHE}/{split}/s{source}/meta.parquet")
    for cat, sel in (("A: no shared term", (sh_m == 0)),
                     ("C: rare shared term but not retrieved",
                      (sh_m > 0) & (df_m <= df_max))):
        ex = sm[sel.nonzero()[0][:6]] if np.any(sel) else []
        print(f"\n  --- examples {cat} ---")
        for i in ex:
            qg = q_global[rows[i]]
            tg = keep[sub_tr[i]]
            r1, r2 = m1.row(int(qg), named=True), m2.row(int(tg), named=True)
            print(f"    S1 name={r1['name_norm']!r}\n       addr={r1['addr_norm']!r}")
            print(f"    S{source} name={r2['name_norm']!r}\n       addr={r2['addr_norm']!r}")
            print(f"       S1 toks={r1['name_toks'].split(chr(9))}|{r1['addr_toks'].split(chr(9))}"
                  f"  nums={r1['nums'].split(chr(9))}")
            print(f"       S{source} toks={r2['name_toks'].split(chr(9))}|{r2['addr_toks'].split(chr(9))}"
                  f"  nums={r2['nums'].split(chr(9))}\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(country=a[0] if a else "India", source=int(a[1]) if len(a) > 1 else 2,
         topk=int(a[2]) if len(a) > 2 else 20)
