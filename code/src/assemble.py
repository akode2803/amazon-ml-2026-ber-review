"""
Final assembly of the routed mixture of experts.

Each expert (infer2.py with COUNTRIES=... DUMP=...) scores only the countries
it is responsible for:

    US      -> stack trained on US only        (in-domain specialist)
    India   -> stack trained on India only     (in-domain specialist)
    France  -> joint US+India stack            (zero-shot: the only model
                                                that has seen both scripts)

Countries never share a target record (country agreement on true pairs is
100%), so the dumps are disjoint and can simply be concatenated.  Then the
same decision layer as validation: mutual exclusivity per target, and the
expected-F0.5 prefix rule with a per-country operating point.

Usage:  python assemble.py dumpA.npz dumpB.npz ...
Env: PARAMS "US:1.1:1.0,India:1.1:1.0,France:1.1:4.0", OUTDIR, SPLIT
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide  # noqa: E402
from infer import mark_taken, write_tsv  # noqa: E402
from model import ROOT  # noqa: E402

CACHE = f"{ROOT}/work/cache"
SPLIT = os.environ.get("SPLIT", "test")
OUTDIR = os.environ.get("OUTDIR", f"{ROOT}/output")
MIN_P = 0.02
PARAMS = {k: (float(t), float(a)) for k, t, a in
          (x.split(":") for x in os.environ.get(
              "PARAMS", "US:1.1:1.0,India:1.1:1.0,France:1.1:4.0").split(","))}


def main(paths):
    os.makedirs(OUTDIR, exist_ok=True)
    meta1 = pl.read_parquet(f"{CACHE}/{SPLIT}/s1/meta.parquet",
                            columns=["id", "country"])
    ids1 = meta1["id"].to_numpy()
    ctry = meta1["country"].to_numpy()
    n1 = len(ids1)
    Z = [np.load(p) for p in paths]
    cat = lambda k: np.concatenate([z[k] for z in Z])  # noqa: E731
    write_tsv(f"{OUTDIR}/candidate_pairs.tsv", "candidate_entity_ids", ids1,
              cat("c_row"), cat("c_tid"), cat("c_src"))
    row, tid, src = cat("row"), cat("tid"), cat("src")
    p = cat("p").astype(np.float64)
    covered = np.zeros(n1, bool); covered[cat("c_row")] = True
    for cn in sorted(set(ctry.tolist())):
        m = ctry == cn
        print(f"  {cn:<8} S1 {m.sum():>9,}  with >=1 candidate "
              f"{covered[m].mean() * 100:6.2f}%")

    key = src.astype(np.int64) * (int(tid.max()) + 2) + tid
    o = np.lexsort((-p, key))
    k2 = key[o]
    best = np.zeros(len(p), bool)
    best[o[np.concatenate([[True], k2[1:] != k2[:-1]])]] = True
    print(f"mutual exclusivity: kept {best.sum():,} of {len(p):,}")
    row, tid, src, p = row[best], tid[best], src[best], p[best]
    order = np.lexsort((-p, row))
    row, tid, src, p = row[order], tid[order], src[order], p[order]
    cnt = np.bincount(row, minlength=n1)
    ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt)])
    take = np.zeros(n1, np.int32)
    for cn in sorted(set(ctry.tolist())):
        ts, ab = PARAMS.get(cn, PARAMS.get("France"))
        t_c = np.zeros(n1, np.int32); v_c = np.zeros(n1, np.float64)
        decide.decide_all(ptr, p, ts, ab, MIN_P, t_c, v_c)
        m = ctry == cn
        take[m] = t_c[m]
        print(f"  {cn:<8} t={ts} a={ab}  mean pred {take[m].mean():.3f}  "
              f"abstain {(take[m] == 0).mean() * 100:5.2f}%")
    sel = np.zeros(len(p), bool)
    mark_taken(ptr, take, sel)
    print(f"decision: {take.sum():,} matches over {n1:,} entities")
    write_tsv(f"{OUTDIR}/matching_results.tsv", "matched_entity_ids", ids1,
              row[sel], tid[sel], src[sel])


if __name__ == "__main__":
    main(sys.argv[1:])
