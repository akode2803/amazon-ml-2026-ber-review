"""
Stage 2b: blocking-term generation, v2.

Rebuilt from the cached parse (no re-parsing) so term-set ablations are cheap.

The v1 term set lost 7.4% of true pairs outright, and 74.8% of those losses
were pairs that *did* share terms -- but only terms with df > 3000.  Two
individually common tokens ("baba", "producer") are jointly rare, so the fix is
df-informed **compound keys**, not a larger top-K: a larger K buys recall at
linear cost in candidates, whereas a compound key buys recall at ~zero cost in
candidates because it is rare by construction.

Term families (each in its own namespace so they can never collide):
    n: name token                    a: address token
    k: name token skeleton           b: address token skeleton
    c: concatenated name             #: building number
    s: sorted concatenated skeleton  d: domain 5-gram
    2: pair of name skeletons        3: pair of address skeletons
    4: number x address skeleton     5: name skeleton x address skeleton
    6: pair of numbers

Compound families pick their members as the *rarest* tokens by a document
frequency computed over S1+S2+S3 within one country, so the selection is
consistent across sources -- which is what makes a compound key actually match.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import skel  # noqa: E402
from prep import h  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"

DEFAULT = dict(simple=True, concat=True, name_pairs=3, addr_pairs=3,
               num_addr=2, cross=2, num_pairs=2, domain=True)


def split_col(s):
    return [x for x in s.split("\t") if x] if s else []


def build_df_table(split, country, cols=("name_toks", "name_skels", "addr_toks",
                                         "addr_skels", "nums")):
    """Document frequency of every token, pooled over S1+S2+S3 of one country.

    Pooling is what makes compound-key selection agree across sources: the
    rarest-token ranking of an S1 record and of its true S2 match are computed
    against the same statistics.
    """
    from collections import Counter
    cnt = Counter()
    for s in (1, 2, 3):
        m = pl.read_parquet(f"{CACHE}/{split}/s{s}/meta.parquet",
                            columns=["country", *cols])
        m = m.filter(pl.col("country") == country)
        for c in cols:
            for v in m[c].to_list():
                if v:
                    cnt.update(set(v.split("\t")) - {""})
        del m
    return cnt


def rarest(toks, dft, k):
    """The k rarest distinct tokens, ties broken lexicographically for determinism."""
    u = sorted(set(t for t in toks if t))
    if len(u) <= k:
        return u
    return sorted(sorted(u, key=lambda t: (dft.get(t, 0), t))[:k])


def record_terms_v2(name_toks, name_skels, addr_toks, addr_skels, nums, dom,
                    legal, dft, cfg):
    t = []
    nt = split_col(name_toks)
    ns = [x for x in split_col(name_skels) if len(x) >= 3]
    at = split_col(addr_toks)
    ab = [x for x in split_col(addr_skels) if len(x) >= 3]
    nm = [x for x in split_col(nums) if len(x) >= 2]
    lg = split_col(legal)
    # a record whose every name token was a legal suffix or filler would
    # otherwise carry no name evidence at all
    if len(nt) < 2 and lg:
        nt = nt + lg
        ns = ns + [skel(x) for x in lg if len(skel(x)) >= 3]

    if cfg["simple"]:
        t += ["n:" + x for x in nt]
        t += ["k:" + x for x in ns]
        t += ["a:" + x for x in at if len(x) >= 3]
        t += ["b:" + x for x in ab]
        t += ["#:" + x for x in nm]
    if cfg["concat"] and len(nt) >= 1:
        cc = "".join(nt)
        if len(cc) >= 6:
            t.append("n:" + cc)                      # 'sk platform' -> 'skplatform'
            t.append("k:" + skel(cc))
        ss = "".join(sorted(ns))
        if len(ss) >= 6:
            t.append("s:" + ss)                      # order-free fingerprint
    if cfg["domain"] and dom and len(dom) >= 7:
        for i in range(len(dom) - 4):
            t.append("d:" + dom[i:i + 5])
    if cfg["name_pairs"]:
        for a, b in combinations(rarest(ns, dft, cfg["name_pairs"]), 2):
            t.append(f"2:{a}|{b}")
    if cfg["addr_pairs"]:
        for a, b in combinations(rarest(ab, dft, cfg["addr_pairs"]), 2):
            t.append(f"3:{a}|{b}")
    if cfg["num_addr"]:
        rn = rarest(nm, dft, cfg["num_addr"])
        ra = rarest(ab, dft, cfg["num_addr"])
        for a in rn:
            for b in ra:
                t.append(f"4:{a}|{b}")
    if cfg["cross"]:
        for a in rarest(ns, dft, cfg["cross"]):
            for b in rarest(ab, dft, cfg["cross"]):
                t.append(f"5:{a}|{b}")
    if cfg["num_pairs"] and len(nm) >= 2:
        for a, b in combinations(rarest(nm, dft, cfg["num_pairs"] + 1), 2):
            t.append(f"6:{a}|{b}")
    return t


def emit(split, countries, cfg=None, tag="v2"):
    cfg = {**DEFAULT, **(cfg or {})}
    for country in countries:
        dft = build_df_table(split, country)
        print(f"  [{split}/{country}] df table: {len(dft):,} distinct tokens",
              flush=True)
        for s in (1, 2, 3):
            m = pl.read_parquet(f"{CACHE}/{split}/s{s}/meta.parquet",
                                columns=["country", "name_toks", "name_skels",
                                         "addr_toks", "addr_skels", "nums",
                                         "dom", "legal"])
            sel = (m["country"] == country).to_numpy().nonzero()[0]
            rows = m[sel]
            tptr, tval = [0], []
            cols = [rows[c].to_list() for c in ("name_toks", "name_skels",
                                               "addr_toks", "addr_skels",
                                               "nums", "dom", "legal")]
            for i in range(len(sel)):
                terms = record_terms_v2(cols[0][i], cols[1][i], cols[2][i],
                                        cols[3][i], cols[4][i], cols[5][i],
                                        cols[6][i], dft, cfg)
                ids = {h(x) for x in terms}
                tval.extend(ids)
                tptr.append(len(tval))
            out = f"{CACHE}/{split}/s{s}"
            np.savez(f"{out}/terms_{tag}_{country}.npz",
                     rows=sel.astype(np.int32),
                     tptr=np.asarray(tptr, np.int64),
                     tval=np.asarray(tval, np.uint32))
            print(f"    s{s} {country}: {len(sel):,} rows, {len(tval):,} terms "
                  f"({len(tval)/max(len(sel),1):.1f}/rec)", flush=True)
            del m, rows, cols


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    cs = sys.argv[2:] or None
    if cs is None:
        cs = sorted(set(pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                                        columns=["country"])["country"].to_list()))
    emit(split, cs)
