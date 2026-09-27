"""
Pick the candidate operating point from the candwide superset (c2all) and
write the production candidate files (c2) in the candgen schema, so that
model.Partition / run_train / stack / infer consume them unchanged.

Rule grammar (all ranks are within the partition, i.e. per source):
    keep = [prod: fwd<F | rev<R]  OR  cheap_rank_q < M  OR  cheap_rank_c < C
    keep &= cheap_p >= PMIN            (drops hopeless pairs before the
                                        expensive featurisation)

    python select_cand.py eval                 # val-fold curves only
    python select_cand.py write M C PMIN F R   # write c2 for train and test
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
import pairs as P  # noqa: E402

CACHE = f"{M.ROOT}/work/cache"
SRC = os.environ.get("SRC", "c2all")
DST = os.environ.get("DST", "c2")


def load(split, country, source):
    return dict(np.load(f"{CACHE}/{split}/cand/{SRC}_{country}_s{source}.npz"))


def labels(z, source, gt):
    s1row, pq, ps, tgt_row = gt
    rows1, rowsT = z["rows1"], z["rowsT"]
    qpos = np.full(int(max(s1row.max(), rows1.max())) + 2, -1, np.int64)
    qpos[rows1] = np.arange(len(rows1))
    locT = np.full(int(max(rowsT.max(), tgt_row.max())) + 2, -1, np.int64)
    locT[rowsT] = np.arange(len(rowsT))
    sel = (ps == source) & (qpos[s1row[pq]] >= 0)
    gq = qpos[s1row[pq[sel]]]
    gc = locT[tgt_row[sel]]
    ok = gc >= 0
    return P.gold_mask(z["q"], z["c"], len(rowsT), gq[ok], gc[ok]), gq[ok]


def mask(z, m_, c_, pmin, f_, r_):
    k = np.zeros(len(z["q"]), bool)
    if f_ or r_:
        k |= (z["frank"] < f_) | (z["rrank"] < r_)
    if m_:
        k |= z["rq"] < m_
    if c_:
        k |= z["rc"] < c_
    if pmin > 0:
        k &= z["pc"] >= pmin
    return k


GRID = []
for f_, r_ in ((20, 5), (12, 5), (0, 0)):
    for m_ in (0, 5, 8, 10, 15):
        for c_ in (0, 3):
            for pmin in (0.0, 1e-4, 1e-3, 3e-3):
                if (f_ or m_ or c_):
                    GRID.append((m_, c_, pmin, f_, r_))


def evaluate():
    split = "train"
    gt = M.gt_arrays(split)
    ids = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet", columns=["id"])["id"].to_numpy()
    fold = M.fold_of(ids)
    n_s1 = int((fold == 0).sum())
    tot = {g: [0, 0] for g in GRID}
    n_true = 0
    for country in M.countries_of(split):
        for source in (2, 3):
            z = load(split, country, source)
            lab, gq = labels(z, source, gt)
            f1 = fold[z["rows1"]]
            isv = f1[z["q"]] == 0
            n_true += int((f1[gq] == 0).sum())
            for g in GRID:
                k = mask(z, *g) & isv
                tot[g][0] += int(k.sum()); tot[g][1] += int((k & lab).sum())
    print(f"val: {n_s1:,} S1, {n_true:,} true pairs")
    print(f"  {'fwd<':>4} {'rev<':>4} {'q<':>3} {'c<':>3} {'pmin':>6} {'cand/S1':>8} {'recall':>8}")
    for g, (nc, nt) in sorted(tot.items(), key=lambda t: t[1][0]):
        m_, c_, pmin, f_, r_ = g
        print(f"  {f_:>4} {r_:>4} {m_:>3} {c_:>3} {pmin:>6g} {nc / n_s1:>8.2f} "
              f"{nt / n_true * 100:>7.3f}%")


def write(m_, c_, pmin, f_, r_):
    for split in ("train", "test"):
        for country in M.countries_of(split):
            for source in (2, 3):
                p = f"{CACHE}/{split}/cand/{SRC}_{country}_s{source}.npz"
                if not os.path.exists(p):
                    print("  missing", p)
                    continue
                z = load(split, country, source)
                k = mask(z, m_, c_, pmin, f_, r_)
                np.savez(f"{CACHE}/{split}/cand/{DST}_{country}_s{source}.npz",
                         rows1=z["rows1"], rowsT=z["rowsT"], q=z["q"][k],
                         c=z["c"][k], s=z["s"][k], frank=z["frank"][k],
                         rrank=z["rrank"][k], pc=z["pc"][k])
                print(f"  {split} {country} S{source}: {k.sum():,} pairs "
                      f"({k.sum() / len(z['rows1']):.1f}/S1)")


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate()
    else:
        a = sys.argv[2:]
        write(int(a[0]), int(a[1]), float(a[2]), int(a[3]), int(a[4]))
