"""
India dense-retrieval rescue, step 3 (local): the tuned retriever's top-DK
neighbours that the lexical candidate set (c2) does not already contain.  They
are exported in the cross-encoder's own text format (normalised name ; address,
as it was trained) for scoring by the large CE on RunPod.

Writes work/runpod/dense_new.parquet (a, b) and dense_new_keys.npz
(split 0=train/1=test, r1, rT, src, sim, rank, y -- y only meaningful on train).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
import xenc as X  # noqa: E402

R = f"{M.ROOT}/work/runpod"
DK = int(os.environ.get("DK", "5"))
COUNTRY = os.environ.get("COUNTRY", "India")
SUF = os.environ.get("SUF", "")                               # _US, _FR
SPLITS = os.environ.get("SPLITS", "train,test").split(",")


def main():
    import polars as pl
    s1row, pq, ps, tgt = M.gt_arrays("train")
    cols = {k: [] for k in ("split", "r1", "rT", "src", "sim", "rank", "y")}
    texts = []
    for split in SPLITS:
        si = 0 if split == "train" else 1
        for s in (2, 3):
            z = np.load(f"{R}/dense2_knn_{split}_s{s}{SUF}.npz")
            m = z["rank"] < DK
            r1, rT = z["s1"][m].astype(np.int64), z["t"][m].astype(np.int64)
            c = np.load(f"{M.CACHE}/{split}/cand/c2_{COUNTRY}_s{s}.npz")
            have = np.unique((c["rows1"][c["q"]].astype(np.int64) << 32)
                             | c["rowsT"][c["c"]].astype(np.int64))
            new = ~np.isin((r1 << 32) | rT, have)
            r1, rT = r1[new], rT[new]
            if split == "train":
                gk = np.unique((s1row[pq[ps == s]].astype(np.int64) << 32)
                               | tgt[ps == s].astype(np.int64))
                y = np.isin((r1 << 32) | rT, gk)
            else:
                y = np.zeros(len(r1), bool)
            src = np.full(len(r1), s, np.int8)
            a, b = X.pair_texts(split, r1, rT, src)
            texts.append(pl.DataFrame({"a": a.tolist(), "b": b.tolist()}))
            for k, v in (("split", np.full(len(r1), si, np.int8)), ("r1", r1), ("rT", rT),
                         ("src", src), ("sim", z["sim"][m][new]), ("rank", z["rank"][m][new]),
                         ("y", y)):
                cols[k].append(v)
            print(f"  {split} s{s}: {len(r1):,} new pairs (top-{DK}), true {y.sum():,}", flush=True)
    pl.concat(texts).write_parquet(f"{R}/dense_new{SUF}.parquet", compression="zstd")
    np.savez(f"{R}/dense_new_keys{SUF}.npz", **{k: np.concatenate(v) for k, v in cols.items()})


if __name__ == "__main__":
    main()
