import os
"""Final score as a function of the (k_fwd, k_rev) candidate operating point.

Uses the cached validation probabilities, so each point costs a filter and a
decision pass rather than a rescoring run.
"""
import os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
z = np.load(f"{ROOT}/work/models/val_scores_c1.npz")
gt, p, pos, src, tid = z["gt"], z["p"], z["pos"], z["src"], z["tid"]
fr, rr = z["fr"], z["rr"]
T, fold, g2gt = z["T"], z["fold"], z["g2gt"]
val_rows = (fold == 0).nonzero()[0]
val_gt = np.sort(g2gt[val_rows])
remap = np.full(len(T), -1, np.int64); remap[val_gt] = np.arange(len(val_gt))
keep0 = remap[gt] >= 0
g0 = remap[gt[keep0]]; p0 = p[keep0]; y0 = pos[keep0].astype(np.int8)
s0 = src[keep0]; t0 = tid[keep0]; f0 = fr[keep0]; r0 = rr[keep0]
Tv = T[val_gt].astype(np.int32)
n = len(val_gt)
print(f"{'k_fwd':>6} {'k_rev':>6} {'cand/S1':>9} {'cand recall':>12} "
      f"{'oracle':>9} {'FINAL F0.5':>11} {'prec':>8} {'rec':>8}")
for kf, kr in [(3,2),(5,2),(5,3),(8,3),(8,5),(12,3),(12,5),(20,3),(20,5)]:
    m = (f0 < kf) | (r0 < kr)
    gg, pp, yy, ss, tt = g0[m], p0[m], y0[m], s0[m], t0[m]
    key = ss.astype(np.int64) * (int(tt.max()) + 2) + tt
    o = np.lexsort((-pp, key)); k2 = key[o]
    best = np.zeros(len(pp), bool)
    best[o[np.concatenate([[True], k2[1:] != k2[:-1]])]] = True
    found = np.zeros(n, np.int32); np.add.at(found, gg[yy.astype(bool)], 1)
    oracle = np.where(Tv == 0, 1.0, np.where(found == 0, 0.0,
                      1.25 * found / np.maximum(found + 0.25 * Tv, 1e-9))).mean()
    gg, pp, yy = gg[best], pp[best], yy[best]
    order = np.lexsort((-pp, gg)); gg, pp, yy = gg[order], pp[order], yy[order]
    cnt = np.bincount(gg, minlength=n)
    ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt)])
    take = np.zeros(n, np.int32); val = np.zeros(n, np.float64)
    decide.decide_all(ptr, pp.astype(np.float64), 1.15, 4.0, 0.02, take, val)
    f05 = decide.f05_from_take(ptr, yy, take, Tv)
    tp = sum(int(yy[ptr[i]:ptr[i]+take[i]].sum()) for i in range(n) if take[i])
    prec = tp / max(int(take.sum()), 1); rec = tp / int(Tv.sum())
    print(f"{kf:>6} {kr:>6} {m.sum()/n:>9.2f} {y0[m].sum()/Tv.sum()*100:>11.3f}% "
          f"{oracle:>9.5f} {f05:>11.5f} {prec:>8.4f} {rec:>8.4f}", flush=True)
