"""
Training pairs for a fine-tuned India retrieval embedder: ground-truth matches
(S1 <-> S2/S3) of India entities outside the val fold, raw text.  Every pair
with non-Latin script on either side (the English <-> native-script cases the
lexical blocker misses) is kept; the rest are sampled.

Writes work/runpod/bienc_train.parquet (a = S1 text, b = target text).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402

DATA = f"{M.ROOT}/student_resource/dataset"
OUT = f"{M.ROOT}/work/runpod"
N_LATIN = int(os.environ.get("N_LATIN", "300000"))
COUNTRY = os.environ.get("BI_COUNTRY", "India")
OUTNAME = os.environ.get("BI_OUT", "bienc_train.parquet")
# France has no labels: its training pairs are the pipeline's own confident test
# matches (pseudo-labels, as in stack.py self-training) -- "France:<dump>:<n>"
PSEUDO = os.environ.get("BI_PSEUDO", "")


def texts(k):
    raw = pl.read_csv(f"{DATA}/train/train_source{k}.tsv", separator="\t", quote_char=None,
                      infer_schema_length=0)
    return (raw["business_name"].fill_null("") + ", "
            + raw["business_address"].fill_null("")).to_numpy()


def pseudo_pairs(spec):
    """Confident test matches of one country: p >= 0.98 and the target's best claimant."""
    country, dump, n = spec.split(":")
    z = np.load(f"{M.ROOT}/work/dumps/{dump}.npz")
    p, row, src, tid = z["p"], z["row"], z["src"], z["tid"].astype(np.int64)
    key = src.astype(np.int64) * (int(tid.max()) + 2) + tid
    o = np.lexsort((-p, key))
    win = np.zeros(len(p), bool)
    win[o[np.r_[True, key[o][1:] != key[o][:-1]]]] = True
    m = win & (p >= 0.98)
    raw = {k: pl.read_csv(f"{DATA}/test/test_source{k}.tsv", separator="\t", quote_char=None,
                          infer_schema_length=0) for k in (1, 2, 3)}
    txt = {k: (r["business_name"].fill_null("") + ", " + r["business_address"].fill_null("")).to_numpy()
           for k, r in raw.items()}
    ids = {k: r["entity_id"].str.slice(3).cast(pl.Int64).to_numpy() for k, r in raw.items()}
    a, b = [], []
    for s in (2, 3):
        ms = m & (src == s)
        o2 = np.argsort(ids[s])
        rT = o2[np.searchsorted(ids[s][o2], tid[ms])]
        a.append(txt[1][row[ms]]); b.append(txt[s][rT])
    a = np.concatenate(a); b = np.concatenate(b)
    sel = np.random.default_rng(11).choice(len(a), min(int(n), len(a)), replace=False)
    print(f"  pseudo {country}: {m.sum():,} confident matches, exported {len(sel):,}", flush=True)
    return a[sel], b[sel]


def main():
    if PSEUDO:
        a, b = pseudo_pairs(PSEUDO)
        pl.DataFrame({"a": a.tolist(), "b": b.tolist()}) \
            .write_parquet(f"{OUT}/{OUTNAME}", compression="zstd")
        return
    s1row, pq, ps, tgt = M.gt_arrays("train")
    meta = pl.read_parquet(f"{M.CACHE}/train/s1/meta.parquet", columns=["id", "country"])
    fold = M.fold_of(meta["id"].to_numpy()); ctry = meta["country"].to_numpy()
    r1 = s1row[pq]
    keep = (fold[r1] != 0) & (ctry[r1] == COUNTRY)
    t1 = texts(1)
    a_all, b_all = [], []
    for s in (2, 3):
        m = keep & (ps == s)
        ts = texts(s)
        a_all.append(t1[r1[m]]); b_all.append(ts[tgt[m]])
    a = np.concatenate(a_all); b = np.concatenate(b_all)
    nonlatin = np.fromiter((any(ord(ch) > 0x24F for ch in x + y) for x, y in zip(a, b)),
                           bool, len(a))
    rng = np.random.default_rng(9)
    lat = np.flatnonzero(~nonlatin)
    sel = np.concatenate([np.flatnonzero(nonlatin),
                          rng.choice(lat, min(N_LATIN, len(lat)), replace=False)])
    rng.shuffle(sel)
    pl.DataFrame({"a": a[sel].tolist(), "b": b[sel].tolist()}) \
        .write_parquet(f"{OUT}/{OUTNAME}", compression="zstd")
    print(f"{COUNTRY} non-val match pairs {len(a):,}; non-Latin {nonlatin.sum():,}; "
          f"exported {len(sel):,}", flush=True)


if __name__ == "__main__":
    main()
