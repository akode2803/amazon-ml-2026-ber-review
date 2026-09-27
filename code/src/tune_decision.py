"""
Decision-layer study, run against cached validation probabilities so it can be
iterated without retraining.

Compares:
  * fixed probability thresholds (the conventional choice)
  * top-1-only
  * the expected-F0.5 greedy rule from metric.py
  * each of the above under a hard **mutual-exclusivity** constraint

Mutual exclusivity is not a heuristic here: in the training ground truth every
one of the 7,638,365 matched S2/S3 records is claimed by exactly one S1 entity.
So when two S1 entities both want the same record, at most one can be right,
and conceding the record to the more confident claimant can only help precision.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
MODELS = f"{ROOT}/work/models"
CTAG = os.environ.get("CTAG", "c1")
VAL_FOLD = int(os.environ.get("VAL_FOLD", "0"))


def stats(ptr, yy, take, Tv):
    f = decide.f05_from_take(ptr, yy, take, Tv)
    tp = 0
    for g in range(len(take)):
        lo = ptr[g]
        if take[g]:
            tp += int(yy[lo:lo + take[g]].sum())
    npred = int(take.sum())
    prec = tp / max(npred, 1)
    rec = tp / max(int(Tv.sum()), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return f, prec, rec, f1, npred, (take == 0).mean()


def main():
    z = np.load(f"{MODELS}/val_scores_{CTAG}.npz")
    gt, p, pos, src, tid = z["gt"], z["p"], z["pos"], z["src"], z["tid"]
    T, fold, g2gt = z["T"], z["fold"], z["g2gt"]
    meta1 = pl.read_parquet(f"{CACHE}/train/s1/meta.parquet",
                            columns=["id", "country"])
    ctry = meta1["country"].to_numpy()
    # entity list = every val-fold entity (including ones with no candidates)
    inv = np.empty(len(T), np.int64)          # GT row -> global S1 row
    inv[g2gt] = np.arange(len(g2gt))
    val_rows = (fold == VAL_FOLD).nonzero()[0]
    val_gt = np.sort(g2gt[val_rows])
    remap = np.full(len(T), -1, np.int64); remap[val_gt] = np.arange(len(val_gt))
    ctry_ent = ctry[inv[val_gt]]

    keep = remap[gt] >= 0
    g = remap[gt[keep]]; pp = p[keep]; yy = pos[keep].astype(np.int8)
    ss = src[keep]; tt = tid[keep]
    Tv = T[val_gt].astype(np.int32)
    print(f"validation: {len(val_gt):,} entities  {len(g):,} pairs  "
          f"{Tv.sum():,} true matches  {(Tv==0).mean()*100:.2f}% singletons")
    print(f"candidate recall here: {yy.sum()/Tv.sum()*100:.3f}%   "
          f"oracle ceiling: {oracle(g, yy, Tv, len(val_gt)):.5f}")

    def prep(mask=None):
        gi, pi, yi = (g, pp, yy) if mask is None else (g[mask], pp[mask], yy[mask])
        o = np.lexsort((-pi, gi))
        gi, pi, yi = gi[o], pi[o], yi[o]
        cnt = np.bincount(gi, minlength=len(val_gt))
        ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt)])
        return ptr, pi, yi

    # ---- mutual exclusivity: a target may only be claimed by its best bidder
    key = ss.astype(np.int64) * (int(tt.max()) + 2) + tt
    o = np.lexsort((-pp, key))
    k2 = key[o]
    firstbid = np.zeros(len(pp), bool)
    firstbid[o[np.concatenate([[True], k2[1:] != k2[:-1]])]] = True
    dup = 1.0 - firstbid.mean()
    print(f"pairs dropped by mutual exclusivity: {dup*100:.2f}%  "
          f"(of those dropped, {yy[~firstbid].mean()*100:.3f}% were true)")

    print(f"\n{'rule':<44} {'F0.5':>8} {'prec':>8} {'rec':>8} {'F1':>8} "
          f"{'pred':>11} {'abstain':>8}")
    rows = []
    for excl, tagx in ((None, ""), (firstbid, " +excl")):
        ptr, pi, yi = prep(excl)
        for thr in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
            take = count_above(ptr, pi, thr)
            s = stats(ptr, yi, take, Tv)
            rows.append((f"threshold {thr}{tagx}", *s))
        for ts in (1.0, 1.05, 1.15, 1.3):
            for ab in (1.0, 2.0, 4.0, 8.0):
                take = np.zeros(len(val_gt), np.int32)
                val = np.zeros(len(val_gt), np.float64)
                decide.decide_all(ptr, pi.astype(np.float64), ts, ab, 0.02,
                                  take, val)
                s = stats(ptr, yi, take, Tv)
                rows.append((f"expF0.5 t={ts} a={ab}{tagx}", *s))
    rows.sort(key=lambda r: -r[1])
    for r in rows[:18]:
        print(f"  {r[0]:<42} {r[1]:>8.5f} {r[2]:>8.4f} {r[3]:>8.4f} "
              f"{r[4]:>8.4f} {r[5]:>11,} {r[6]*100:>7.2f}%")
    print("  ...")
    for r in rows[-6:]:
        print(f"  {r[0]:<42} {r[1]:>8.5f} {r[2]:>8.4f} {r[3]:>8.4f} "
              f"{r[4]:>8.4f} {r[5]:>11,} {r[6]*100:>7.2f}%")

    # ---- best config, broken down
    best = rows[0]
    print(f"\nBEST: {best[0]}  macro-F0.5 = {best[1]:.5f}")
    ptr, pi, yi = prep(firstbid if "+excl" in best[0] else None)
    ts, ab = parse_cfg(best[0])
    take = np.zeros(len(val_gt), np.int32); val = np.zeros(len(val_gt), np.float64)
    if ts is None:
        take = count_above(ptr, pi, float(best[0].split()[1]))
    else:
        decide.decide_all(ptr, pi.astype(np.float64), ts, ab, 0.02, take, val)
    breakdown(ptr, pi, yi, Tv, take, ctry_ent)


def parse_cfg(name):
    if not name.startswith("expF0.5"):
        return None, None
    parts = dict(x.split("=") for x in name.replace("+excl", "").split()[1:])
    return float(parts["t"]), float(parts["a"])


def count_above(ptr, pi, thr):
    take = np.zeros(len(ptr) - 1, np.int32)
    ok = (pi >= thr).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(ok)])
    take[:] = cs[ptr[1:]] - cs[ptr[:-1]]
    return take


def oracle(g, yy, Tv, n):
    found = np.zeros(n, np.int32)
    np.add.at(found, g[yy.astype(bool)], 1)
    return float(np.where(Tv == 0, 1.0, np.where(found == 0, 0.0,
                 1.25 * found / np.maximum(found + 0.25 * Tv, 1e-9))).mean())


def breakdown(ptr, pi, yi, Tv, take, ctry_ent):
    n = len(take)
    per = np.zeros(n)
    c = np.zeros(n, np.int32)
    for gi in range(n):
        lo = ptr[gi]; k = take[gi]
        cc = int(yi[lo:lo + k].sum()) if k else 0
        c[gi] = cc
        if k == 0:
            per[gi] = 1.0 if Tv[gi] == 0 else 0.0
        elif Tv[gi] == 0 or cc == 0:
            per[gi] = 0.0
        else:
            per[gi] = 1.25 * cc / (0.25 * Tv[gi] + k)
    print(f"\n  by country:")
    for cn in sorted(set(ctry_ent.tolist())):
        m = ctry_ent == cn
        print(f"    {cn:<8} n={m.sum():>8,}  F0.5={per[m].mean():.5f}  "
              f"abstain={(take[m]==0).mean()*100:5.2f}%")
    print(f"\n  by true match count T:")
    for t in range(0, 8):
        m = Tv == t
        if m.sum():
            print(f"    T={t}  n={m.sum():>8,}  F0.5={per[m].mean():.5f}  "
                  f"mean pred={take[m].mean():.2f}  mean correct={c[m].mean():.2f}")
    m = Tv >= 8
    if m.sum():
        print(f"    T>=8 n={m.sum():>8,}  F0.5={per[m].mean():.5f}")
    print(f"\n  worst buckets (entities scoring 0 with T>0): "
          f"{((per==0)&(Tv>0)).sum():,} = {((per==0)&(Tv>0)).mean()*100:.2f}%")
    print(f"  singletons predicted non-empty (false merges): "
          f"{((Tv==0)&(take>0)).sum():,} of {(Tv==0).sum():,} "
          f"= {((Tv==0)&(take>0)).sum()/max((Tv==0).sum(),1)*100:.2f}%")


if __name__ == "__main__":
    main()
