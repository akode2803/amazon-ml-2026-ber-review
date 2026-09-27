"""
Evaluate a persisted candidate set on train.

The headline number is the **oracle ceiling**: the macro-F0.5 a perfect
matching model would score given these candidates.  It is the hard upper bound
blocking imposes on the final result, and it is what makes the
recall-vs-candidate-count trade-off concrete -- pair recall alone does not,
because the metric is macro-averaged per entity and punishes an entity that
loses one of its three matches far more than the pair count suggests.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"


def load(split, out_tag, countries=None):
    meta1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet", columns=["country"])
    countries = countries or sorted(set(meta1["country"].to_list()))
    return countries


def main(split="train", out_tag="c1"):
    g = np.load(f"{CACHE}/gt_{split}.npz")
    s1row, pq, ps, tgt_row = g["s1row"], g["pq"], g["ps"], g["tgt_row"]
    n_gt = len(s1row)
    n_s1rows = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                               columns=["id"]).height
    g2gt = np.full(n_s1rows, -1, np.int64)
    g2gt[s1row] = np.arange(n_gt)
    c1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                         columns=["country"])["country"].to_numpy()
    countries = sorted(set(c1.tolist()))

    T = np.zeros(n_gt, np.int32)            # true matches per entity
    np.add.at(T, pq, 1)
    found = np.zeros(n_gt, np.int32)
    ncand = np.zeros(n_gt, np.int32)
    found_fwd = np.zeros(n_gt, np.int32)
    found_rev = np.zeros(n_gt, np.int32)
    tot_pairs = 0
    pos_scores, neg_scores = [], []

    for country in countries:
        for source in (2, 3):
            f = f"{CACHE}/{split}/cand/{out_tag}_{country}_s{source}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f)
            rows1, rowsT, q, c, s, prov = (z["rows1"], z["rowsT"], z["q"],
                                           z["c"], z["s"], z["prov"])
            nT = len(rowsT)
            gt_of_slot = g2gt[rows1]
            np.add.at(ncand, gt_of_slot, np.bincount(q, minlength=len(rows1)))
            tot_pairs += len(q)

            # true pairs in this partition, as (slot, local target)
            qpos = np.full(n_s1rows, -1, np.int64); qpos[rows1] = np.arange(len(rows1))
            locT = np.full(int(rowsT.max()) + 2, -1, np.int64)
            locT[rowsT] = np.arange(nT)
            sel = (ps == source) & (qpos[s1row[pq]] >= 0)
            tslot = qpos[s1row[pq[sel]]]
            tloc = locT[tgt_row[sel]]
            gtix = pq[sel]
            ok = tloc >= 0
            tslot, tloc, gtix = tslot[ok], tloc[ok], gtix[ok]

            gold = np.sort(tslot.astype(np.int64) * nT + tloc)
            key = q.astype(np.int64) * nT + c
            j = np.searchsorted(gold, key)
            j = np.clip(j, 0, max(len(gold) - 1, 0))
            is_pos = (len(gold) > 0) & (gold[j] == key)

            gkey = tslot.astype(np.int64) * nT + tloc
            kk = np.sort(key)
            jj = np.searchsorted(kk, gkey)
            jj = np.clip(jj, 0, max(len(kk) - 1, 0))
            hit = (len(kk) > 0) & (kk[jj] == gkey)
            np.add.at(found, gtix, hit.astype(np.int32))

            # provenance of the hits
            pm = {}
            for a in range(len(q)):
                pass
            order = np.argsort(key)
            skey, sprov = key[order], prov[order]
            jj2 = np.searchsorted(skey, gkey)
            jj2 = np.clip(jj2, 0, max(len(skey) - 1, 0))
            okp = (len(skey) > 0) & (skey[jj2] == gkey)
            pv = np.where(okp, sprov[jj2], -1)
            np.add.at(found_fwd, gtix, ((pv == 0) | (pv == 2)).astype(np.int32))
            np.add.at(found_rev, gtix, ((pv == 1) | (pv == 2)).astype(np.int32))

            pos_scores.append(s[is_pos])
            neg_scores.append(s[~is_pos][::37])      # subsample negatives
            del z, rows1, rowsT, q, c, s, prov, qpos, locT

    pos = np.concatenate(pos_scores); neg = np.concatenate(neg_scores)
    rec = found.sum() / T.sum()
    # oracle: predict exactly the true matches that survived blocking
    f05 = np.where(T == 0, 1.0,
                   np.where(found == 0, 0.0,
                            1.25 * found / (found + 0.25 * T)))
    print(f"\n=== candidate set '{out_tag}' on {split} ===")
    print(f"  total candidate pairs      {tot_pairs:,}")
    print(f"  candidates / S1  mean {ncand.mean():.2f}  median {np.median(ncand):.0f}"
          f"  p95 {np.percentile(ncand,95):.0f}  p99 {np.percentile(ncand,99):.0f}"
          f"  max {ncand.max()}")
    print(f"  S1 with 0 candidates       {(ncand==0).mean()*100:.3f}%")
    print(f"  pair recall                {rec*100:.3f}%  ({found.sum():,}/{T.sum():,})")
    print(f"  entities fully covered     {(found==T).mean()*100:.3f}%")
    print(f"  entities with 0 of >0      {((found==0)&(T>0)).mean()*100:.3f}%")
    print(f"  ORACLE macro-F0.5 ceiling  {f05.mean():.5f}")
    print(f"  recall via forward only    {found_fwd.sum()/T.sum()*100:.3f}%")
    print(f"  recall via reverse pass    {found_rev.sum()/T.sum()*100:.3f}%")
    print(f"  rescued only by reverse    {(found_rev.sum()-(found_fwd.sum()+found_rev.sum()-found.sum()))/T.sum()*100:.3f}%")
    print(f"\n  block score  positives: mean {pos.mean():.4f} "
          f"p05 {np.percentile(pos,5):.4f} p25 {np.percentile(pos,25):.4f} "
          f"median {np.median(pos):.4f}")
    print(f"               negatives: mean {neg.mean():.4f} "
          f"p50 {np.median(neg):.4f} p95 {np.percentile(neg,95):.4f} "
          f"p99 {np.percentile(neg,99):.4f}")
    print("\n  effect of a min-score cut (on this candidate set):")
    print(f"  {'cut':>8} {'pairs kept':>12} {'pair recall':>12} {'cands/S1':>9}")
    for cut in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30):
        kp = (pos >= cut).sum() / len(pos)
        frac = ((pos >= cut).sum() + (neg >= cut).sum() * 37) / (len(pos) + len(neg) * 37)
        print(f"  {cut:>8.2f} {frac*100:>11.2f}% {rec*kp*100:>11.3f}% "
              f"{ncand.mean()*frac:>9.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train",
         sys.argv[2] if len(sys.argv) > 2 else "c1")
