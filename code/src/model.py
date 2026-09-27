"""
Stages 4-6: pair scoring, calibration, and the expected-F0.5 decision layer.

Training negatives are the candidate set itself.  That matters: the pairs the
blocker retrieves and rejects *are* the hard negatives -- same street, same
franchise name, same city, one digit apart -- so the classifier is trained on
exactly the distribution it will see at inference.  Sampling random non-matches
instead would make the task trivially easy and the probabilities useless for a
precision-first decision rule.

The decision layer does not threshold.  It maximises the expected value of the
competition's own per-entity metric (see metric.py), which makes the effective
acceptance bar rise as an entity accumulates confident matches, and makes
abstention a first-class outcome rather than a fallback.
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
import features as F  # noqa: E402
import pairs as P  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"


# ------------------------------------------------------------------ utilities
def idf_tables(split, country):
    f = f"{CACHE}/{split}/idf_{country}.pkl"
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return pickle.load(fh)
    t = F.build_idf_tables(split, country)
    with open(f, "wb") as fh:
        pickle.dump(t, fh, protocol=4)
    return t


def gt_arrays(split):
    g = np.load(f"{CACHE}/gt_{split}.npz")
    return g["s1row"], g["pq"], g["ps"], g["tgt_row"]


def fold_of(ids, nfold=5):
    """Deterministic entity-level fold from the numeric id (not row order)."""
    x = (ids.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(13)
    return (x % np.uint64(nfold)).astype(np.int8)


class Partition:
    """Everything needed to featurize one (country, source) candidate file."""

    def __init__(self, split, country, source, out_tag, idft,
                 k_fwd=9999, k_rev=9999):
        z = np.load(f"{CACHE}/{split}/cand/{out_tag}_{country}_s{source}.npz")
        self.rows1, self.rowsT = z["rows1"], z["rowsT"]
        q, c, s = z["q"], z["c"], z["s"]
        fr, rr = z["frank"], z["rrank"]
        # every (k_fwd, k_rev) operating point is a filter over one generation
        keep = (fr < k_fwd) | (rr < k_rev)
        if not keep.all():
            q, c, s, fr, rr = q[keep], c[keep], s[keep], fr[keep], rr[keep]
        self.q, self.c, self.s, self.frank, self.rrank = q, c, s, fr, rr
        self.country, self.source = country, source
        self.qs = F.FieldStore(split, country, 1, idft)
        self.cs = F.FieldStore(split, country, source, idft)
        assert np.array_equal(self.qs.rows, self.rows1)
        assert np.array_equal(self.cs.rows, self.rowsT)
        self.ctx = P.context(self.q, self.c, self.s, source, self.qs,
                             self.cs, self.frank, self.rrank)

    def labels(self, s1row, pq, ps, tgt_row):
        n_s1rows = int(s1row.max()) + 2
        qpos = np.full(max(n_s1rows, int(self.rows1.max()) + 2), -1, np.int64)
        qpos[self.rows1] = np.arange(len(self.rows1))
        locT = np.full(int(self.rowsT.max()) + 2, -1, np.int64)
        locT[self.rowsT] = np.arange(len(self.rowsT))
        sel = (ps == self.source) & (qpos[s1row[pq]] >= 0)
        gq = qpos[s1row[pq[sel]]]
        gc = locT[tgt_row[sel]]
        ok = gc >= 0
        return P.gold_mask(self.q, self.c, len(self.rowsT), gq[ok], gc[ok]), pq[sel][ok]

    def feats(self, lo, hi):
        sl = slice(lo, hi)
        return F.compute(self.qs, self.cs, self.q[sl], self.c[sl], self.ctx[sl])

    def free(self):
        del self.qs, self.cs
        gc.collect()


def countries_of(split):
    return sorted(set(pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                                      columns=["country"])["country"].to_list()))


# ------------------------------------------------------------------- training
def build_training_set(split, out_tag, keep_folds, max_pairs_per_part,
                       countries=None, seed=0):
    s1row, pq, ps, tgt_row = gt_arrays(split)
    ids1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                           columns=["id"])["id"].to_numpy()
    fold = fold_of(ids1)
    rng = np.random.default_rng(seed)
    Xs, ys, gs, cs = [], [], [], []
    for country in (countries or countries_of(split)):
        idft = idf_tables(split, country)
        for source in (2, 3):
            t0 = time.time()
            part = Partition(split, country, source, out_tag, idft)
            lab, _ = part.labels(s1row, pq, ps, tgt_row)
            qfold = fold[part.rows1][part.q]
            sel = np.isin(qfold, keep_folds)
            idx = sel.nonzero()[0]
            if len(idx) > max_pairs_per_part:
                idx = np.sort(rng.choice(idx, max_pairs_per_part, replace=False))
            # featurize the selected pairs in slices to bound peak memory
            Xp = np.empty((len(idx), len(F.feature_names())), np.float32)
            step = 2_000_000
            for a in range(0, len(idx), step):
                b = min(a + step, len(idx))
                sub = idx[a:b]
                Xp[a:b] = F.compute(part.qs, part.cs, part.q[sub], part.c[sub],
                                    part.ctx[sub])
            Xs.append(Xp); ys.append(lab[idx].astype(np.int8))
            gs.append(part.rows1[part.q[idx]].astype(np.int32))
            cs.append(np.full(len(idx), f"{country}", dtype=object))
            print(f"    train rows {country} S{source}: {len(idx):,} "
                  f"({lab[idx].mean()*100:.2f}% pos) {time.time()-t0:.0f}s", flush=True)
            part.free(); del part
            gc.collect()
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs),
            np.concatenate(cs))


# Similarity features whose effect must be non-decreasing: a pair can never
# become *less* likely to match because its names/addresses agree more.
# Robustness prior for unseen countries; ranks and context stay free.
_MONO_UP_SUFFIX = ("_inter", "_jac", "_cont", "_widf", "_wjac", "_mxsh")
_MONO_UP = {s[0] for s in F.STR_SCORERS} | {"ad_eq", "adl_eq", "nm_eq"}


def mono_vector(names, prefix=""):
    """+1 for similarity features (names optionally carry `prefix`), else 0."""
    out = []
    for n in names:
        if prefix:
            if not n.startswith(prefix):
                out.append(0)
                continue
            n = n[len(prefix):]
        up = n in _MONO_UP or (n.split("_")[0] in F.SHORT
                               and n.endswith(_MONO_UP_SUFFIX))
        out.append(1 if up else 0)
    return out


def train_lgbm(X, y, params=None, num_round=600, valid=None):
    import lightgbm as lgb
    env = os.environ.get
    p = dict(objective="binary", metric=["binary_logloss", "auc"],
             learning_rate=float(env("LGB_LR", "0.05")),
             num_leaves=int(env("LGB_LEAVES", "255")),
             min_data_in_leaf=int(env("LGB_MINLEAF", "100")),
             feature_fraction=float(env("LGB_FF", "0.8")),
             bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=float(env("LGB_L2", "1.0")),
             num_threads=int(env("LGB_THREADS", "20")), verbosity=-1, max_bin=127)
    if env("LGB_MONO", "0") == "1":
        p.update(monotone_constraints=mono_vector(F.feature_names()),
                 monotone_constraints_method="intermediate")
    p.update(params or {})
    ds = lgb.Dataset(X, label=y, feature_name=F.feature_names())
    vs = [lgb.Dataset(valid[0], label=valid[1], reference=ds)] if valid else None
    cb = ([lgb.log_evaluation(100), lgb.early_stopping(50, verbose=True)]
          if valid else [])
    return lgb.train(p, ds, num_boost_round=num_round, valid_sets=vs, callbacks=cb)
