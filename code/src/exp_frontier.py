"""
The candidate-generation Pareto frontier.

Because each stored pair carries its forward rank and its reverse rank, every
(k_fwd, k_rev) configuration is a filter over one generation run.  For each we
report pair recall, candidates per S1, and the **oracle macro-F0.5** -- the
score a perfect matcher would get on that candidate set, which is the quantity
the leaderboard is actually bounded by.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
NORANK = 9999


def main(split="train", out_tag="c1"):
    g = np.load(f"{CACHE}/gt_{split}.npz")
    s1row, pq, ps, tgt_row = g["s1row"], g["pq"], g["ps"], g["tgt_row"]
    n_gt = len(s1row)
    meta1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet", columns=["id", "country"])
    n_s1rows = meta1.height
    g2gt = np.full(n_s1rows, -1, np.int64); g2gt[s1row] = np.arange(n_gt)
    countries = sorted(set(meta1["country"].to_list()))
    T = np.zeros(n_gt, np.int32); np.add.at(T, pq, 1)

    # gather all partitions once, in memory-light form
    blocks = []
    for country in countries:
        for source in (2, 3):
            f = f"{CACHE}/{split}/cand/{out_tag}_{country}_s{source}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f)
            rows1, rowsT = z["rows1"], z["rowsT"]
            q, c, fr, rr = z["q"], z["c"], z["frank"], z["rrank"]
            nT = len(rowsT)
            qpos = np.full(n_s1rows, -1, np.int64); qpos[rows1] = np.arange(len(rows1))
            locT = np.full(int(rowsT.max()) + 2, -1, np.int64)
            locT[rowsT] = np.arange(nT)
            sel = (ps == source) & (qpos[s1row[pq]] >= 0)
            gk = qpos[s1row[pq[sel]]] * nT + locT[tgt_row[sel]]
            gtix = pq[sel]
            ok = locT[tgt_row[sel]] >= 0
            gk, gtix = np.sort(gk[ok]), gtix[ok]
            key = q.astype(np.int64) * nT + c
            j = np.searchsorted(gk, key); j = np.clip(j, 0, max(len(gk) - 1, 0))
            is_pos = (len(gk) > 0) & (gk[j] == key)
            # the GT row of a pair is just its own S1 entity's GT row
            blocks.append(dict(gt_of_slot=g2gt[rows1], q=q, fr=fr, rr=rr,
                               is_pos=is_pos, n1=len(rows1)))
            del z, rows1, rowsT, q, c, fr, rr, qpos, locT

    print(f"{'k_fwd':>6} {'k_rev':>6} {'pairs':>14} {'cand/S1':>9} "
          f"{'pair recall':>12} {'ent-full':>9} {'oracle F0.5':>12}")
    rows = []
    for kf in (0, 2, 3, 5, 8, 12, 20):
        for kr in (0, 1, 2, 3, 5):
            if kf == 0 and kr == 0:
                continue
            found = np.zeros(n_gt, np.int32)
            ncand = np.zeros(n_gt, np.int32)
            tot = 0
            for b in blocks:
                m = (b["fr"] < kf) | (b["rr"] < kr)
                tot += int(m.sum())
                np.add.at(ncand, b["gt_of_slot"], np.bincount(
                    b["q"][m], minlength=b["n1"]))
                hit = m & b["is_pos"]
                np.add.at(found, b["gt_of_slot"][b["q"][hit]], 1)
            rec = found.sum() / T.sum()
            f05 = np.where(T == 0, 1.0, np.where(found == 0, 0.0,
                           1.25 * found / np.maximum(found + 0.25 * T, 1e-9)))
            rows.append((kf, kr, tot, ncand.mean(), rec, (found == T).mean(),
                         f05.mean()))
            print(f"{kf:>6} {kr:>6} {tot:>14,} {ncand.mean():>9.2f} "
                  f"{rec*100:>11.3f}% {(found==T).mean()*100:>8.2f}% "
                  f"{f05.mean():>12.5f}", flush=True)

    # Pareto frontier on (candidates/S1 down, oracle F0.5 up)
    print("\nPareto-efficient operating points (min cand/S1 for each ceiling):")
    rows.sort(key=lambda r: r[3])
    best = -1
    for kf, kr, tot, cps, rec, ef, f in rows:
        if f > best:
            best = f
            print(f"  k_fwd={kf:<3} k_rev={kr:<2} -> {cps:6.2f} cand/S1, "
                  f"recall {rec*100:6.3f}%, oracle F0.5 {f:.5f}, {tot:,} pairs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train",
         sys.argv[2] if len(sys.argv) > 2 else "c1")
