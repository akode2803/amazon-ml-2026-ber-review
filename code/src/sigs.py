"""
Record-level information content of an address / name *as a whole*.

Token IDF says how rare each word is; it cannot say that the full address
"2100 old hickory rd" is held by exactly one business in the country while
"1 main st" is held by forty.  That distinction is exactly what decides the
"name replaced, address identical" family: a replaced name at a *unique*
address is still the same business, at a *shared* address it is a neighbour.

Per (split, country) we hash three signatures for every S1/S2/S3 record

    ad   exact street identity: sorted numbers + sorted street tokens
    adl  loose street identity: largest number + longest street token
         (survives a dropped / reordered component)
    nm   sorted name content tokens (franchise / chain detector)

and count how many records carry each signature, overall and in S1 only.
The S1 count is the number of distinct reference entities at that address,
i.e. how many rivals a candidate could belong to.  All statistics are
unsupervised and computed from the split's own records, so they exist
identically at test time (France included) and cannot leak labels.

Output, row-aligned with features.FieldStore:
    cache/<split>/sig_<country>_s<source>.npz
        ad, adl, nm                     uint64 signature hashes (0 = empty)
        ad_all, ad_s1, adl_all, adl_s1, nm_all, nm_s1   int32 counts
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"


def _sigs(nums_col, toks_col, name_col):
    """Signature strings per record (python: list.eval multi-key sort is unreliable)."""
    ad, adl, nm = [], [], []
    for nu, to, na in zip(nums_col, toks_col, name_col):
        n = [x for x in nu.split("	") if x] if nu else []
        t = [x for x in to.split("	") if x] if to else []
        if n or t:
            ad.append(" ".join(sorted(n)) + "|" + " ".join(sorted(t)))
            bn = max(n, key=lambda x: (len(x), x)) if n else ""
            lt = min(t, key=lambda x: (-len(x), x)) if t else ""
            adl.append(bn + "|" + lt)
        else:
            ad.append("")
            adl.append("")
        nm.append(" ".join(sorted(x for x in na.split("	") if x)) if na else "")
    return ad, adl, nm


def build(split, country):
    t0 = time.time()
    frames = []
    for s in (1, 2, 3):
        m = (pl.scan_parquet(f"{CACHE}/{split}/s{s}/meta.parquet")
             .filter(pl.col("country") == country)
             .select("nums", "addr_toks", "name_toks").collect())
        ad, adl, nm = _sigs(m["nums"].to_list(), m["addr_toks"].to_list(),
                            m["name_toks"].to_list())
        frames.append(pl.DataFrame({"ad": ad, "adl": adl, "nm": nm,
                                    "src": np.full(len(ad), s, np.int8)}))
        del m
    allm = pl.concat(frames)
    hashed = {}
    for k in ("ad", "adl", "nm"):
        h = allm[k].hash(seed=17, seed_1=1, seed_2=2, seed_3=3).to_numpy()
        h = np.where(allm[k].to_numpy() == "", np.uint64(0), h).astype(np.uint64)
        hashed[k] = h
    src = allm["src"].to_numpy()
    counts = {}
    for k, h in hashed.items():
        u, inv, c_all = np.unique(h, return_inverse=True, return_counts=True)
        c_s1 = np.bincount(inv[src == 1], minlength=len(u))
        empty = u == 0
        c_all[empty] = 0
        c_s1[empty] = 0
        counts[k] = (c_all[inv].astype(np.int32), c_s1[inv].astype(np.int32))
    off = 0
    for s, fr in zip((1, 2, 3), frames):
        n = len(fr)
        sl = slice(off, off + n)
        np.savez(f"{CACHE}/{split}/sig_{country}_s{s}.npz",
                 ad=hashed["ad"][sl], adl=hashed["adl"][sl], nm=hashed["nm"][sl],
                 ad_all=counts["ad"][0][sl], ad_s1=counts["ad"][1][sl],
                 adl_all=counts["adl"][0][sl], adl_s1=counts["adl"][1][sl],
                 nm_all=counts["nm"][0][sl], nm_s1=counts["nm"][1][sl])
        off += n
    a_all = counts["ad"][0]
    print(f"  sig {split}/{country}: {len(allm):,} records, "
          f"unique-address share {(a_all == 1).mean() * 100:.1f}%, "
          f"median addr multiplicity {np.median(a_all[a_all > 0]):.0f}  "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    cs = sys.argv[2:] or sorted(set(pl.read_parquet(
        f"{CACHE}/{split}/s1/meta.parquet", columns=["country"])["country"].to_list()))
    for c in cs:
        build(split, c)
