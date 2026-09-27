"""
Stage 7 (two-stage version): score the test split with the cross-fitted
stage-1 ensemble + stage-2 local-graph model, apply mutual exclusivity and
the expected-F0.5 decision, and write both submission TSVs.

Identical code path to the validation run in stack.py: the test split is
treated exactly like the validation fold (every pair gets the mean of the
four stage-1 models), so the stage-2 inputs have the distribution the
validation number was measured on.

Env: TAG (model dir under work/stack), TSCALE, ABSTAIN, OUTDIR, SPLIT.
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
import model as M  # noqa: E402
import stack as S  # noqa: E402
from infer import mark_taken, write_tsv  # noqa: E402

ROOT = S.ROOT
CACHE = S.CACHE
OUTDIR = os.environ.get("OUTDIR", f"{ROOT}/output")
SPLIT = os.environ.get("SPLIT", "test")
TSCALE = float(os.environ.get("TSCALE", "1.15"))
ABSTAIN = float(os.environ.get("ABSTAIN", "4.0"))
MIN_P = float(os.environ.get("DUMP_MIN_P", "0.02"))  # lower -> dump pairs for a CE rescue band
EXCL = os.environ.get("EXCL", "1") == "1"
DUMP = os.environ.get("DUMP", "")
OOD_COUNTRIES = [c for c in os.environ.get("OOD_COUNTRIES", "France").split(",") if c]
TSCALE_OOD = float(os.environ.get("TSCALE_OOD", str(TSCALE)))
ABSTAIN_OOD = float(os.environ.get("ABSTAIN_OOD", str(ABSTAIN)))


def main():
    t00 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    import lightgbm as lgb
    models = S.load_stage1()
    b2 = lgb.Booster(model_file=f"{S.WORK}/stage2.txt")
    with open(f"{S.WORK}/iso2.pkl", "rb") as fh:
        iso2 = pickle.load(fh)
    meta1 = pl.read_parquet(f"{CACHE}/{SPLIT}/s1/meta.parquet",
                            columns=["id", "country"])
    ids1 = meta1["id"].to_numpy()
    n1 = len(ids1)
    countries = [c for c in M.countries_of(SPLIT)
                 if not S.COUNTRIES or c in S.COUNTRIES]
    print(f"{SPLIT}: {n1:,} S1 entities, countries {countries}  "
          f"model {S.TAG}  t={TSCALE} a={ABSTAIN}", flush=True)

    acc = {k: [] for k in ("row", "tid", "src", "p")}
    cand = {k: [] for k in ("row", "tid", "src")}
    for country in countries:
        t0 = time.time()
        D, stores = S.score_country(SPLIT, country, models)
        X2 = S.s2_features(D, stores)
        kept = D["keep"].nonzero()[0]
        p2 = np.zeros(len(D["q"]), np.float32)
        p2[kept] = iso2.predict(b2.predict(X2, num_iteration=b2.best_iteration))
        del X2
        row = D["rows1"][D["q"]].astype(np.int32)
        tids = {s: pl.read_parquet(f"{CACHE}/{SPLIT}/s{s}/meta.parquet",
                                   columns=["id"])["id"].to_numpy()[stores[s][0]]
                for s in (2, 3)}
        tid = np.empty(len(row), np.int32)
        for s in (2, 3):
            m = D["src"] == s
            tid[m] = tids[s][D["c"][m]]
        cand["row"].append(row); cand["tid"].append(tid); cand["src"].append(D["src"])
        k = p2 >= MIN_P
        acc["row"].append(row[k]); acc["tid"].append(tid[k])
        acc["src"].append(D["src"][k]); acc["p"].append(p2[k])
        print(f"  {country}: {len(row):,} candidate pairs, {k.sum():,} with "
              f"p2>={MIN_P}, mean p2 on kept {p2[kept].mean():.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        del D, stores, p2; gc.collect()

    if DUMP:   # per-expert partial result; assemble.py merges and decides
        np.savez(DUMP, c_row=np.concatenate(cand["row"]),
                 c_tid=np.concatenate(cand["tid"]), c_src=np.concatenate(cand["src"]),
                 row=np.concatenate(acc["row"]), tid=np.concatenate(acc["tid"]),
                 src=np.concatenate(acc["src"]), p=np.concatenate(acc["p"]))
        print(f"dumped -> {DUMP}  total {time.time() - t00:.0f}s")
        return
    write_tsv(f"{OUTDIR}/candidate_pairs.tsv", "candidate_entity_ids", ids1,
              np.concatenate(cand["row"]), np.concatenate(cand["tid"]),
              np.concatenate(cand["src"]))
    del cand; gc.collect()

    row = np.concatenate(acc["row"]); tid = np.concatenate(acc["tid"])
    src = np.concatenate(acc["src"]); p = np.concatenate(acc["p"]).astype(np.float64)
    if EXCL:
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
    take = np.zeros(n1, np.int32); val = np.zeros(n1, np.float64)
    decide.decide_all(ptr, p, TSCALE, ABSTAIN, MIN_P, take, val)
    ctry = meta1["country"].to_numpy()
    # zero-shot countries (France) get their own operating point: the LOCO
    # proxy showed a Latin-script unseen country loses *precision*, so a
    # higher abstention weight pays there while hurting nowhere else
    if OOD_COUNTRIES:
        take2 = np.zeros(n1, np.int32); val2 = np.zeros(n1, np.float64)
        decide.decide_all(ptr, p, TSCALE_OOD, ABSTAIN_OOD, MIN_P, take2, val2)
        ood = np.isin(ctry, OOD_COUNTRIES)
        take[ood] = take2[ood]
        print(f"OOD countries {OOD_COUNTRIES}: t={TSCALE_OOD} a={ABSTAIN_OOD}")
    sel = np.zeros(len(p), bool)
    mark_taken(ptr, take, sel)
    for cn in countries:
        m = ctry == cn
        print(f"  {cn:<8} entities {m.sum():>9,}  mean pred {take[m].mean():.3f}  "
              f"abstain {(take[m] == 0).mean() * 100:5.2f}%")
    print(f"decision: {take.sum():,} matches, abstain {(take == 0).mean() * 100:.2f}%")
    write_tsv(f"{OUTDIR}/matching_results.tsv", "matched_entity_ids", ids1,
              row[sel], tid[sel], src[sel])
    print(f"total {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
