"""
Stage 7: inference on the test split -> output/candidate_pairs.tsv and
output/matching_results.tsv.

France appears only in the test set.  Nothing in the pipeline is keyed on the
country label: normalisation is script-driven, idf is computed per country from
that country's own test records, and every feature is a ratio, a rank or an
idf-weighted quantity, so the France partition is handled by exactly the same
code path with weights derived from France's own term statistics.
"""
from __future__ import annotations

import gc
import os
import pickle
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide  # noqa: E402
import features as F  # noqa: E402
import model as M  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
MODELS = f"{ROOT}/work/models"
OUTDIR = os.environ.get("OUTDIR", f"{ROOT}/student_resource/output")

SPLIT = os.environ.get("SPLIT", "test")
CTAG = os.environ.get("CTAG", "c1")
K_FWD = int(os.environ.get("K_FWD", "12"))
K_REV = int(os.environ.get("K_REV", "5"))
TSCALE = float(os.environ.get("TSCALE", "1.15"))
ABSTAIN = float(os.environ.get("ABSTAIN", "4.0"))
MIN_P = float(os.environ.get("MIN_P", "0.02"))
EXCL = os.environ.get("EXCL", "1") == "1"


def main():
    t00 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    import lightgbm as lgb
    booster = lgb.Booster(model_file=f"{MODELS}/lgbm_{CTAG}.txt")
    with open(f"{MODELS}/iso_{CTAG}.pkl", "rb") as fh:
        iso = pickle.load(fh)
    nit = booster.best_iteration or booster.num_trees()

    meta1 = pl.read_parquet(f"{CACHE}/{SPLIT}/s1/meta.parquet",
                            columns=["id", "country"])
    ids1 = meta1["id"].to_numpy()
    n1 = len(ids1)
    countries = sorted(set(meta1["country"].to_list()))
    print(f"{SPLIT}: {n1:,} S1 entities, countries {countries}", flush=True)

    # accumulate per-pair results across partitions
    acc_row, acc_tid, acc_src, acc_p = [], [], [], []
    cand_row, cand_tid, cand_src = [], [], []
    for country in countries:
        idft = M.idf_tables(SPLIT, country)
        for source in (2, 3):
            t0 = time.time()
            part = M.Partition(SPLIT, country, source, CTAG, idft,
                               k_fwd=K_FWD, k_rev=K_REV)
            tgt_ids = pl.read_parquet(f"{CACHE}/{SPLIT}/s{source}/meta.parquet",
                                      columns=["id"])["id"].to_numpy()
            grow = part.rows1[part.q].astype(np.int32)
            tid = tgt_ids[part.rowsT][part.c].astype(np.int32)
            probs = np.empty(len(part.q), np.float32)
            for a in range(0, len(part.q), 2_000_000):
                b = min(a + 2_000_000, len(part.q))
                X = F.compute(part.qs, part.cs, part.q[a:b], part.c[a:b],
                              part.ctx[a:b])
                probs[a:b] = iso.predict(booster.predict(X, num_iteration=nit))
                del X
            cand_row.append(grow); cand_tid.append(tid)
            cand_src.append(np.full(len(tid), source, np.int8))
            keep = probs >= MIN_P
            acc_row.append(grow[keep]); acc_tid.append(tid[keep])
            acc_src.append(np.full(int(keep.sum()), source, np.int8))
            acc_p.append(probs[keep])
            print(f"  scored {country:7s} S{source}: {len(part.q):,} pairs, "
                  f"{keep.sum():,} above min_p  {time.time()-t0:.0f}s", flush=True)
            part.free(); del part, probs, grow, tid
            F._HMEMO.clear(); gc.collect()

    # ------------------------------------------------- candidate_pairs.tsv
    write_tsv(f"{OUTDIR}/candidate_pairs.tsv", "candidate_entity_ids", ids1,
              np.concatenate(cand_row), np.concatenate(cand_tid),
              np.concatenate(cand_src))
    del cand_row, cand_tid, cand_src; gc.collect()

    # ------------------------------------------------- decision
    row = np.concatenate(acc_row); tid = np.concatenate(acc_tid)
    src = np.concatenate(acc_src); p = np.concatenate(acc_p)
    del acc_row, acc_tid, acc_src, acc_p; gc.collect()
    # --- mutual exclusivity: the ground truth is many-to-one (every matched
    # S2/S3 record in train belongs to exactly one S1 entity), so a record may
    # only be kept by whichever entity is most confident about it.
    if EXCL:
        key = src.astype(np.int64) * (int(tid.max()) + 2) + tid
        o = np.lexsort((-p, key))
        k2 = key[o]
        best = np.zeros(len(p), bool)
        best[o[np.concatenate([[True], k2[1:] != k2[:-1]])]] = True
        print(f"mutual exclusivity: kept {best.sum():,} of {len(p):,} scored "
              f"pairs ({(1-best.mean())*100:.1f}% conceded to a rival entity)")
        row, tid, src, p = row[best], tid[best], src[best], p[best]

    order = np.lexsort((-p, row))
    row, tid, src, p = row[order], tid[order], src[order], p[order]
    cnt = np.bincount(row, minlength=n1)
    ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt)])
    take = np.zeros(n1, np.int32); val = np.zeros(n1, np.float64)
    decide.decide_all(ptr, p.astype(np.float64), TSCALE, ABSTAIN, MIN_P,
                      take, val)
    sel = np.zeros(len(p), bool)
    mark_taken(ptr, take, sel)
    print(f"\ndecision: {take.sum():,} predicted matches over {n1:,} entities "
          f"({(take==0).mean()*100:.2f}% abstain, mean {take.mean():.2f}/entity)")
    write_tsv(f"{OUTDIR}/matching_results.tsv", "matched_entity_ids", ids1,
              row[sel], tid[sel], src[sel])
    print(f"total {time.time()-t00:.0f}s")


import numba  # noqa: E402


@numba.njit(cache=True)
def mark_taken(ptr, take, sel):
    for g in range(ptr.shape[0] - 1):
        for j in range(take[g]):
            sel[ptr[g] + j] = True


def write_tsv(path, col, ids1, row, tid, src):
    """One row per S1 entity, comma-joined ID list, empty when none.

    Written in entity blocks: materialising all ~90M id strings at once would
    cost several GB as a numpy object/unicode array, so each block builds only
    its own strings and streams them out.
    """
    n1 = len(ids1)
    o = np.argsort(row, kind="stable")
    row, tid, src = row[o], tid[o], src[o]
    del o
    cnt = np.bincount(row, minlength=n1)
    ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt)])
    nonempty = int((cnt > 0).sum())
    BLK = 200_000
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for lo in range(0, n1, BLK):
            hi = min(lo + BLK, n1)
            a, b = ptr[lo], ptr[hi]
            pre = np.where(src[a:b] == 2, "S2-", "S3-")
            ids = np.char.add(pre, tid[a:b].astype("U10"))
            out = []
            for i in range(lo, hi):
                s0, s1_ = ptr[i] - a, ptr[i + 1] - a
                out.append(f"S1-{ids1[i]}\t"
                           + (",".join(ids[s0:s1_]) if s1_ > s0 else "") + "\n")
            fh.write("".join(out))
            del pre, ids, out
    print(f"  wrote {path}: {n1:,} rows, {nonempty:,} non-empty, "
          f"{len(row):,} ids")


if __name__ == "__main__":
    main()
