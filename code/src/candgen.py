"""
Stage 3 (production): generate and persist the candidate set.

Two retrieval directions, unioned:

  forward  S1 -> S2/S3 : ranks candidates for each reference entity.
  reverse  S2/S3 -> S1 : rescues *asymmetric* pairs.  A candidate whose address
                         is empty and whose name is a 2-token fragment has very
                         little evidence, so from S1's side it sits far down a
                         crowded ranking -- but from its own side the true S1
                         entity is its clear best match.  The reverse pass adds
                         at most k_rev pairs per candidate record, so it costs
                         few candidates per S1 while recovering exactly the
                         pairs the forward pass structurally cannot rank.

Both directions score with the same IDF-weighted cosine, so their scores are
comparable and can be max-combined in the union.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"


def pairs_from_topk(cand, score, cnt, nq):
    """Dense (nq, k) top-k arrays -> flat (qslot, cloc, score, rank) tuples."""
    k = cand.shape[1]
    q = np.repeat(np.arange(nq, dtype=np.int32), cnt.astype(np.int64))
    mask = np.arange(k)[None, :] < cnt[:, None]
    rank = np.broadcast_to(np.arange(k, dtype=np.int16)[None, :], cand.shape)[mask]
    return (q, cand[mask].astype(np.int32), score[mask].astype(np.float32),
            rank.astype(np.int16))


NORANK = np.int16(9999)


def union_pairs(qa, ca, sa, ra, qb, cb, sb, rb, nq, nc):
    """Union two directed pair lists, keeping each direction's rank.

    Storing both ranks (rather than a provenance flag) makes the whole
    (k_fwd, k_rev) Pareto frontier derivable from a single generation run by
    filtering, instead of re-running retrieval per configuration.
    """
    allk = np.concatenate([qa.astype(np.int64) * nc + ca,
                           qb.astype(np.int64) * nc + cb])
    alls = np.concatenate([sa, sb])
    fr = np.concatenate([ra, np.full(len(qb), NORANK, np.int16)])
    rr = np.concatenate([np.full(len(qa), NORANK, np.int16), rb])
    o = np.argsort(allk, kind="stable")
    allk, alls, fr, rr = allk[o], alls[o], fr[o], rr[o]
    first = np.ones(len(allk), bool)
    first[1:] = allk[1:] != allk[:-1]
    gid = np.cumsum(first) - 1
    ng = gid[-1] + 1 if len(gid) else 0
    smax = np.zeros(ng, np.float32); np.maximum.at(smax, gid, alls)
    fmin = np.full(ng, NORANK, np.int16); np.minimum.at(fmin, gid, fr)
    rmin = np.full(ng, NORANK, np.int16); np.minimum.at(rmin, gid, rr)
    k = allk[first]
    return ((k // nc).astype(np.int32), (k % nc).astype(np.int32),
            smax, fmin, rmin)


def run(split, tag="v2", k_fwd=20, k_rev=3, ac=1.0, df_max=3000, budget=4000,
        min_score=0.0, do_reverse=True, out_tag="c1", countries=None):
    os.makedirs(f"{CACHE}/{split}/cand", exist_ok=True)
    meta1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet", columns=["country"])
    countries = countries or sorted(set(meta1["country"].to_list()))
    for country in countries:
        rows1, ptr1, val1 = block.load_terms(split, 1, country, tag)
        n1 = len(rows1)
        idx1 = block.Index(ptr1, val1, n1) if do_reverse else None
        for source in (2, 3):
            t0 = time.time()
            rowsT, ptrT, valT = block.load_terms(split, source, country, tag)
            nT = len(rowsT)
            idxT = block.Index(ptrT, valT, nT)
            tb = time.time() - t0

            c, s, n, touch = block.query_index(
                idxT, ptr1, val1, df_max=df_max, budget=budget, topk=k_fwd,
                aq=1.0, ac=ac, min_score=min_score)
            qf, cf, sf, rf = pairs_from_topk(c, s, n, n1)
            tf = time.time() - t0
            del c, s, n, touch

            z32, z16 = np.zeros(0, np.int32), np.zeros(0, np.int16)
            qr = cr = z32; sr = np.zeros(0, np.float32); rr = z16
            if do_reverse:
                c, s, n, touch = block.query_index(
                    idx1, ptrT, valT, df_max=df_max, budget=budget, topk=k_rev,
                    aq=1.0, ac=ac, min_score=min_score)
                cr, qr, sr, rr = pairs_from_topk(c, s, n, nT)
                del c, s, n, touch
            tr = time.time() - t0

            q, cc, sc, fmin, rmin = union_pairs(qf, cf, sf, rf, qr, cr, sr, rr,
                                                n1, nT)
            np.savez(f"{CACHE}/{split}/cand/{out_tag}_{country}_s{source}.npz",
                     rows1=rows1, rowsT=rowsT, q=q, c=cc, s=sc,
                     frank=fmin, rrank=rmin)
            print(f"  {country:7s} S{source}: fwd {len(qf):,} + rev {len(qr):,} "
                  f"-> union {len(q):,} pairs ({len(q)/n1:.1f}/S1) | "
                  f"build {tb:.0f}s fwd {tf-tb:.0f}s rev {tr-tf:.0f}s", flush=True)
            del idxT, rowsT, ptrT, valT, qf, cf, sf, rf, qr, cr, sr, rr, q, cc, sc
            gc.collect()
        del idx1, rows1, ptr1, val1
        gc.collect()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--k-fwd", type=int, default=20)
    ap.add_argument("--k-rev", type=int, default=3)
    ap.add_argument("--ac", type=float, default=1.0)
    ap.add_argument("--df-max", type=int, default=3000)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--no-reverse", action="store_true")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out", default="c1")
    a = ap.parse_args()
    run(a.split, tag=a.tag, k_fwd=a.k_fwd, k_rev=a.k_rev, ac=a.ac,
        df_max=a.df_max, budget=a.budget, min_score=a.min_score,
        do_reverse=not a.no_reverse, out_tag=a.out)
