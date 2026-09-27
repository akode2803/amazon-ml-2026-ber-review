"""
Stage 1-2: normalize every record once and cache it in a compact, reloadable
form, partitioned by (country, source).

Why cache: normalization is the only genuinely per-record cost in the pipeline
and it is pure, so it runs once for train+test and is reused by every
blocking / feature / ablation experiment.

Layout under work/cache/<split>/<source>/:
    meta.parquet   id, country, name_norm, addr_norm, name_toks, addr_toks,
                   nums, geo, name_skels, addr_skels, dom   (strings, tab-joined)
    terms.npz      CSR of blocking-term hashes: tptr (int64), tval (uint32)

Terms are 32-bit hashes of "<type>:<string>" folded into 2^26 buckets. Hash
collisions can only *add* spurious candidates, which the scoring stage removes;
they cannot lose a true pair. That lets us skip a 10M-entry python vocabulary
dict and its ~1.5 GB of overhead.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import parse_addr, parse_name, skel  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
DATA = f"{ROOT}/student_resource/dataset"
CACHE = f"{ROOT}/work/cache"

NBUCKET_BITS = 26
NBUCKET = 1 << NBUCKET_BITS
MASK = NBUCKET - 1

# 64-bit FNV-1a: stable across processes and runs (python's hash() is salted).
_FNV_OFF = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def h(s: str) -> int:
    x = _FNV_OFF
    for b in s.encode("utf-8"):
        x = ((x ^ b) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return x & MASK


def record_terms(nm, ad) -> list:
    """Blocking terms for one record. Type prefixes keep namespaces disjoint."""
    t = []
    for x in nm.toks:
        t.append("n:" + x)
    for x in nm.skels:
        if len(x) >= 3:
            t.append("k:" + x)
    if nm.dom and len(nm.dom) >= 6:  # concatenated domain: no word boundaries
        d = nm.dom
        for i in range(len(d) - 4):
            t.append("d:" + d[i:i + 5])
    for x in ad.toks:
        if len(x) >= 3:
            t.append("a:" + x)
    for x in ad.skels:
        if len(x) >= 3:
            t.append("b:" + x)
    for x in ad.nums:
        if len(x) >= 2:
            t.append("#:" + x)
    return t


def _work(chunk):
    names, addrs = chunk
    out_meta, tptr, tval = [], [0], []
    for nraw, araw in zip(names, addrs):
        nm = parse_name(nraw)
        ad = parse_addr(araw)
        terms = record_terms(nm, ad)
        ids = {h(x) for x in terms}
        tval.extend(ids)
        tptr.append(len(tval))
        out_meta.append((
            nm.norm, ad.norm, "\t".join(nm.toks), "\t".join(ad.toks),
            "\t".join(ad.nums), "\t".join(ad.geo), "\t".join(nm.skels),
            "\t".join(ad.skels), nm.dom, "\t".join(nm.legal), ad.tail,
        ))
    return out_meta, np.asarray(tptr, dtype=np.int64), np.asarray(tval, dtype=np.uint32)


COLS = ["name_norm", "addr_norm", "name_toks", "addr_toks", "nums", "geo",
        "name_skels", "addr_skels", "dom", "legal", "tail"]


def prep(split: str, source: int, pool: Pool, chunk: int = 40000) -> None:
    path = f"{DATA}/{split}/{split}_source{source}.tsv"
    t0 = time.time()
    df = pl.read_csv(path, separator="\t", quote_char=None,
                     schema_overrides={c: pl.Utf8 for c in
                                       ["entity_id", "business_name",
                                        "business_address", "country"]})
    n = len(df)
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, n, chunk)]
    res = pool.map(_work, chunks, chunksize=1)

    metas, ptrs, vals = [], [], []
    off = 0
    for m, p, v in res:
        metas.extend(m)
        ptrs.append(p[1:] + off)
        off += len(v)
        vals.append(v)
    tptr = np.concatenate([np.zeros(1, np.int64)] + ptrs)
    tval = np.concatenate(vals)

    out = f"{CACHE}/{split}/s{source}"
    os.makedirs(out, exist_ok=True)
    meta = pl.DataFrame({
        "id": df["entity_id"].str.slice(3).cast(pl.Int32),
        "country": df["country"],
        **{c: [m[i] for m in metas] for i, c in enumerate(COLS)},
    })
    meta.write_parquet(f"{out}/meta.parquet", compression="zstd")
    np.savez(f"{out}/terms.npz", tptr=tptr, tval=tval)
    print(f"  {split}/s{source}: {n:,} rows, {len(tval):,} terms "
          f"({len(tval)/n:.1f}/rec), {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["train", "test"]
    with Pool(16) as pool:
        for split in which:
            for s in (1, 2, 3):
                prep(split, s, pool)
