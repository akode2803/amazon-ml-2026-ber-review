"""
Ablation study.

Builds one modest train matrix and one validation matrix (kept whole, with
entity grouping intact so the decision layer can run), caches both, then
re-trains over feature subsets.  Every ablation therefore differs *only* in the
component under test.

Also runs leave-one-country-out: train on US, validate on India, and vice
versa.  The test set contains France, which has no training labels at all, so
cross-country transfer is the only honest evidence we have about how the model
will behave there.
"""
from __future__ import annotations

import gc
import os
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
OUT = f"{ROOT}/work/models"
SPLIT, CTAG, VAL_FOLD = "train", "c1", 0
TRAIN_CAP = int(os.environ.get("ABL_TRAIN_CAP", "500000"))
VAL_ENT = int(os.environ.get("ABL_VAL_ENT", "120000"))
NROUND = int(os.environ.get("ABL_NROUND", "300"))

NAMES = F.feature_names()
IX = {n: i for i, n in enumerate(NAMES)}


def group(pred):
    return [i for n, i in IX.items() if pred(n)]


GROUPS = {
    "set_name": group(lambda n: n.startswith(("nt_", "ns_"))),
    "set_addr": group(lambda n: n.startswith(("at_", "as_", "nm_", "geo_"))),
    "string_fuzzy": [IX[n] for n in ("nm_tsr", "nm_tso", "nm_rat", "nm_par",
                                     "nm_jw", "sk_rat", "sk_tso", "ad_tsr",
                                     "ad_rat", "ad_par", "tl_rat")],
    "ctx_block": [IX[n] for n in ("blk_score", "blk_rank", "blk_rel",
                                  "blk_margin", "n_cand")],
    "ctx_compete": [IX[n] for n in ("tgt_best", "tgt_rel", "tgt_rank")],
    "ctx_misc": [IX[n] for n in ("src_is3", "q_hasaddr", "c_hasaddr",
                                 "both_addr", "len_ratio_nm", "len_ratio_ad")],
    "retrieval_rank": [IX[n] for n in ("fwd_rank", "rev_rank")],
    "idf_weighted": group(lambda n: n.endswith(("_widf", "_wjac", "_mxsh",
                                                "_mxq", "_mxc"))),
    "skeleton": group(lambda n: n.startswith(("ns_", "as_")) or n.startswith("sk_")),
}


def build_cache():
    """Materialise a train matrix and a grouped validation matrix, once."""
    ftr = f"{CACHE}/abl_train.npz"
    fva = f"{CACHE}/abl_val.npz"
    if os.path.exists(ftr) and os.path.exists(fva):
        return ftr, fva
    s1row, pq, ps, tgt_row = M.gt_arrays(SPLIT)
    n_gt = len(s1row)
    meta1 = pl.read_parquet(f"{CACHE}/{SPLIT}/s1/meta.parquet",
                            columns=["id", "country"])
    ids1, ctry1 = meta1["id"].to_numpy(), meta1["country"].to_numpy()
    fold = M.fold_of(ids1)
    g2gt = np.full(len(ids1), -1, np.int64); g2gt[s1row] = np.arange(n_gt)
    T = np.zeros(n_gt, np.int32); np.add.at(T, pq, 1)
    countries = M.countries_of(SPLIT)

    # fixed validation entity subsample (same entities for every ablation)
    rng = np.random.default_rng(11)
    val_rows = (fold == VAL_FOLD).nonzero()[0]
    val_rows = np.sort(rng.choice(val_rows, min(VAL_ENT, len(val_rows)),
                                  replace=False))
    in_val = np.zeros(len(ids1), bool); in_val[val_rows] = True

    TX, TY, TC = [], [], []
    VX, VY, VG, VC = [], [], [], []
    for country in countries:
        idft = M.idf_tables(SPLIT, country)
        for source in (2, 3):
            t0 = time.time()
            part = M.Partition(SPLIT, country, source, CTAG, idft)
            lab, _ = part.labels(s1row, pq, ps, tgt_row)
            grow = part.rows1[part.q]                    # global S1 row per pair
            qfold = fold[grow]
            tr = np.isin(qfold, [f for f in range(5) if f != VAL_FOLD]).nonzero()[0]
            if len(tr) > TRAIN_CAP:
                tr = np.sort(np.random.default_rng(5).choice(tr, TRAIN_CAP,
                                                             replace=False))
            va = in_val[grow].nonzero()[0]
            for idx, (Xs, Ys, Cs) in ((tr, (TX, TY, TC)), (va, (VX, VY, VC))):
                Xp = np.empty((len(idx), len(NAMES)), np.float32)
                for a in range(0, len(idx), 1_000_000):
                    b = min(a + 1_000_000, len(idx))
                    sub = idx[a:b]
                    Xp[a:b] = F.compute(part.qs, part.cs, part.q[sub],
                                        part.c[sub], part.ctx[sub])
                Xs.append(Xp); Ys.append(lab[idx].astype(np.int8))
                Cs.append(np.full(len(idx), countries.index(country), np.int8))
            VG.append(g2gt[grow[va]].astype(np.int64))
            print(f"  {country:7s} S{source}: train {len(tr):,} val {len(va):,} "
                  f"{time.time()-t0:.0f}s", flush=True)
            part.free(); del part, lab
            F._HMEMO.clear(); gc.collect()
    np.savez(ftr, X=np.concatenate(TX), y=np.concatenate(TY),
             c=np.concatenate(TC))
    del TX, TY, TC; gc.collect()
    np.savez(fva, X=np.concatenate(VX), y=np.concatenate(VY),
             g=np.concatenate(VG), c=np.concatenate(VC),
             T=T, val_rows=val_rows, g2gt=g2gt, ctry=(ctry1 == countries[0]))
    return ftr, fva


def run_decision(prob, g, y, T, tscale=1.05, ab=1.0):
    order = np.lexsort((-prob, g))
    gg, pp, yy = g[order], prob[order], y[order]
    uniq, inv = np.unique(gg, return_inverse=True)
    cnt = np.bincount(inv)
    ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt)])
    Tv = T[uniq].astype(np.int32)
    take = np.zeros(len(uniq), np.int32); val = np.zeros(len(uniq), np.float64)
    decide.decide_all(ptr, pp.astype(np.float64), tscale, ab, 0.02, take, val)
    f_exp = decide.f05_from_take(ptr, yy.astype(np.int8), take, Tv)
    best_thr = max((decide.f05_threshold(ptr, pp, yy.astype(np.int8), Tv, t), t)
                   for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    return f_exp, best_thr, ptr, pp, yy, Tv, take


def main():
    ftr, fva = build_cache()
    tr = np.load(ftr); va = np.load(fva)
    Xtr, ytr, ctr = tr["X"], tr["y"], tr["c"]
    Xva, yva, gva, cva = va["X"], va["y"], va["g"], va["c"]
    T, val_rows = va["T"], va["val_rows"]
    # entities present in the validation subsample but with no candidates at all
    # still need to be scored (they count in the macro average)
    print(f"train {Xtr.shape}  val {Xva.shape}  "
          f"val entities {len(np.unique(gva)):,}")

    def fit_eval(cols, label, train_mask=None, val_mask=None):
        xt = Xtr[:, cols] if train_mask is None else Xtr[train_mask][:, cols]
        yt = ytr if train_mask is None else ytr[train_mask]
        sub = [NAMES[i] for i in cols]
        import lightgbm as lgb
        p = dict(objective="binary", learning_rate=0.08, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.9,
                 bagging_fraction=0.8, bagging_freq=1, num_threads=10,
                 verbosity=-1, max_bin=127)
        b = lgb.train(p, lgb.Dataset(xt, label=yt, feature_name=sub),
                      num_boost_round=NROUND)
        xv = Xva[:, cols]; yv = yva; gv = gva
        if val_mask is not None:
            xv, yv, gv = xv[val_mask], yva[val_mask], gva[val_mask]
        pr = b.predict(xv).astype(np.float32)
        f_exp, (f_thr, thr), *_ = run_decision(pr, gv, yv, T)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(yv, pr)
        print(f"  {label:<34} F0.5(exp)={f_exp:.5f}  F0.5(best-thr)={f_thr:.5f} "
              f"@{thr}  AP={ap:.4f}  ({len(cols)} feats)")
        return f_exp, f_thr, ap

    allc = list(range(len(NAMES)))
    print("\n=== feature-group ablation (LightGBM, expected-F0.5 decision) ===")
    base = fit_eval(allc, "ALL FEATURES")
    for gname, cols in GROUPS.items():
        rest = [i for i in allc if i not in set(cols)]
        f = fit_eval(rest, f"-- without {gname}")
        print(f"     delta vs all: F0.5 {f[0]-base[0]:+.5f}")
    print("\n=== reduced baselines ===")
    fit_eval([IX["blk_score"]], "blocking score only")
    fit_eval([IX["blk_score"], IX["nm_tsr"], IX["ad_tsr"]],
             "naive: tfidf + 2 fuzzy ratios")
    fit_eval([i for i in allc if not NAMES[i].startswith("tgt_")],
             "no many-to-one competition feats")

    print("\n=== leave-one-country-out (proxy for zero-shot France) ===")
    for k, cname in enumerate(M.countries_of(SPLIT)):
        fit_eval(allc, f"train !={cname} -> val {cname}",
                 train_mask=(ctr != k), val_mask=(cva == k))
        fit_eval(allc, f"train ALL      -> val {cname}", val_mask=(cva == k))


if __name__ == "__main__":
    main()
