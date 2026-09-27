"""
Export text pairs for training / scoring a large cross-encoder off-box (RunPod).

Writes to work/runpod/:
    train_ce5.parquet   a, b, y      -- train_pairs4.npz (train pairs + test pseudo pairs)
    score_ce5.parquet   a, b         -- every pair of every all-pairs band, deduplicated
    score_keys.npy      int64 key per score row (split, source, S1 row, target row)
The remote script returns one logit per score row; import_ce.py maps them back
onto each band file, so xenc.py blend can use them like any local CE.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xenc as X  # noqa: E402

OUT = f"{X.ROOT}/work/runpod"
BANDS = [("train", "val_band_all_s2xP3"), ("train", "val_band_all_s2xUS_P2"),
         ("train", "val_band_all_s2xIN_P2"), ("test", "test_band_all_fr_p3"),
         ("test", "test_band_all_us_p2"), ("test", "test_band_all_in_p2")]


def key(split, src, r1, rT):
    return ((np.int64(split == "test") << 62) | (src.astype(np.int64) << 58)
            | (r1.astype(np.int64) << 30) | rT.astype(np.int64))


def main():
    os.makedirs(OUT, exist_ok=True)
    ttag = os.environ.get("TRAIN_TAG", "4")
    z = np.load(f"{X.OUT}/train_pairs{ttag}.npz")
    a = np.empty(len(z["y"]), object); b = np.empty(len(z["y"]), object)
    for k, split in ((0, "train"), (1, "test")):
        m = z["split"] == k
        a[m], b[m] = X.pair_texts(split, z["r1"][m], z["rT"][m], z["src"][m])
    tname = os.environ.get("TRAIN_OUT", "train_ce5.parquet")
    pl.DataFrame({"a": a.tolist(), "b": b.tolist(), "y": z["y"].astype(np.int8)}) \
        .write_parquet(f"{OUT}/{tname}", compression="zstd")
    print(f"train pairs {len(a):,} -> {tname}", flush=True)
    del a, b
    if os.environ.get("SKIP_SCORE", "0") == "1":
        return

    keys, parts = [], []
    for split in ("train", "test"):
        ks = []
        for s, name in BANDS:
            if s == split:
                t = np.load(f"{X.OUT}/{name}.npz")
                ks.append(key(split, t["src"], t["r1"], t["rT"]))
        u = np.unique(np.concatenate(ks))
        src = ((u >> 58) & 0xF).astype(np.int8)
        r1 = ((u >> 30) & ((1 << 28) - 1)).astype(np.int64)
        rT = (u & ((1 << 30) - 1)).astype(np.int64)
        a, b = X.pair_texts(split, r1, rT, src)
        parts.append(pl.DataFrame({"a": a.tolist(), "b": b.tolist()}))
        keys.append(u)
        print(f"score pairs {split}: {len(u):,} unique of {sum(len(k) for k in ks):,}", flush=True)
        del a, b
    pl.concat(parts).write_parquet(f"{OUT}/score_ce5.parquet", compression="zstd")
    np.save(f"{OUT}/score_keys.npy", np.concatenate(keys))


if __name__ == "__main__":
    main()
