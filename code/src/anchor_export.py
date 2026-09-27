"""
Transitive cross-encoder, step 1 (local).  An S1 entity usually has several
records across S2/S3; once one of them is matched confidently (the "anchor"),
a still-uncertain candidate can be checked against that record as well as
against the S1 record -- two views of the same business are often closer to
each other than either is to S1 (different script, missing address, ...).

For every band pair (p2 >= 0.02) of each model, the anchor is the entity's most
confident *other* pair by the current stage-3 score, if that score >= 0.5.
The (candidate record, anchor record) texts are deduplicated and exported for
the large CE; anchor_map_<band>.npy maps each band pair to its row (-1 = none).
No new candidates: only pairs already in the candidate set are re-scored.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xenc as X  # noqa: E402

R = f"{X.ROOT}/work/runpod"
S3 = "s3_c5"
JOBS = [("train", "s2xP3", "fr_p3"), ("train", "s2xUS_P2", "us_p2"), ("train", "s2xIN_P2", "in_p2"),
        ("test", "s2xP3", "fr_p3"), ("test", "s2xUS_P2", "us_p2"), ("test", "s2xIN_P2", "in_p2")]


def anchors(ent, p):
    """Index of each pair's anchor (entity's best other pair with p >= 0.5), -1 if none."""
    o = np.lexsort((-p, ent))
    es = ent[o]
    first = np.r_[True, es[1:] != es[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(es)), 0))
    _, inv, size = np.unique(es, return_inverse=True, return_counts=True)
    best = o[start]
    second = np.where(size[inv] > 1, o[np.minimum(start + 1, len(o) - 1)], -1)
    anc = np.where(o == best, second, best)
    ok = (anc >= 0) & (p[np.maximum(anc, 0)] >= 0.5)
    out = np.full(len(p), -1, np.int64)
    out[o] = np.where(ok, anc, -1)
    return out


def main():
    keys_all, maps = [], {}
    for split, tag, dump in JOBS:
        if split == "train":
            v = np.load(f"{X.ROOT}/work/models/val_scores_{tag}{S3}.npz")
            idx = np.load(f"{X.OUT}/val_band_all_{tag}.npz")["idx"]
            ent, p, src, tid = v["gt"][idx], v["p2"][idx], v["src"][idx], v["tid"][idx]
            band = f"val_band_all_{tag}"
        else:
            z = np.load(f"{X.DUMPD}/{dump}{S3}.npz")
            idx = np.load(f"{X.OUT}/test_band_all_{dump}.npz")["idx"]
            ent, p, src, tid = z["row"][idx], z["p"][idx], z["src"][idx], z["tid"][idx]
            band = f"test_band_all_{dump}"
        anc = anchors(ent.astype(np.int64), p.astype(np.float64))
        rT = np.empty(len(tid), np.int64)
        for s in (2, 3):
            m = src == s
            rT[m] = X.rows_of(split, s, tid[m])
        has = anc >= 0
        a_src, a_row = src[np.maximum(anc, 0)], rT[np.maximum(anc, 0)]
        sp = np.int64(split == "test")
        k = ((sp << 62) | (src.astype(np.int64) << 60) | (rT << 32)
             | (a_src.astype(np.int64) << 29) | a_row)
        k = np.where(has, k, -1)
        maps[band] = k
        keys_all.append(k[has])
        print(f"  {band}: {len(k):,} pairs, {has.sum():,} with a confident anchor", flush=True)
    u = np.unique(np.concatenate(keys_all))
    parts = []
    for sp_, split in ((0, "train"), (1, "test")):
        uu = u[(u >> 62) == sp_]
        src = ((uu >> 60) & 3).astype(np.int8)
        rT = ((uu >> 32) & ((1 << 28) - 1)).astype(np.int64)
        a_src = ((uu >> 29) & 3).astype(np.int8)
        a_row = (uu & ((1 << 29) - 1)).astype(np.int64)
        TT = {s: X.texts(split, s) for s in (2, 3)}
        a = np.where(src == 2, TT[2][np.minimum(rT, len(TT[2]) - 1)],
                     TT[3][np.minimum(rT, len(TT[3]) - 1)])
        b = np.where(a_src == 2, TT[2][np.minimum(a_row, len(TT[2]) - 1)],
                     TT[3][np.minimum(a_row, len(TT[3]) - 1)])
        parts.append(pl.DataFrame({"a": a.tolist(), "b": b.tolist()}))
        del TT
    pl.concat(parts).write_parquet(f"{R}/anchor_pairs.parquet", compression="zstd")
    np.save(f"{R}/anchor_keys.npy", u)
    for band, k in maps.items():
        j = np.searchsorted(u, k)
        np.save(f"{R}/anchor_map_{band}.npy", np.where(k >= 0, j, -1))
    print(f"unique anchor pairs to score: {len(u):,}", flush=True)


if __name__ == "__main__":
    main()
