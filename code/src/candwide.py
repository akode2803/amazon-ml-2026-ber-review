"""
Candidate generation v3: retrieve WIDE, re-rank CHEAP, keep NARROW.

Measured (exp_miss.py, val fold): the true pairs that c1 loses are almost
never "no shared evidence" (<1%).  They are *touched* by retrieval but ranked
20..200 by the idf^2 cosine, because a long S1 address shares rare locality
tokens with dozens of neighbouring businesses while the true record's name is
transliterated.  A bigger top-k buys them back only at linear candidate cost;
a better *ranking* buys them back at ~zero cost.

    1. wide pool     forward top-KF (budget 40k, df_max 30k) + reverse top-KR
    2. cheap score   LightGBM on the 54 numba set features (0.06 s / M pairs),
                     retrieval score/ranks and record-signature features;
                     cross-fitted on entity folds {1,2} / {3,4}, fold 0 (val)
                     and test get the mean -- never an in-sample score
    3. keep          per S1: top-M by cheap score, per target: top-R,
                     optionally union the production rule (fwd<20 | rev<5).
                     The superset (rank_q < 40 | rank_c < 5 | production) is
                     saved with its cheap ranks so the final operating point
                     is a filter, not a re-run.

Retrieval optimises P(true pair in candidate set); the pair model downstream
optimises P(match).  The cheap score is only ever used to *select*.

Usage:
    python candwide.py train      # pools, cheap models, curves, c2all files
    python candwide.py test
Env: BER_ROOT, KF, KR, OUT (default c2)
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numba
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block  # noqa: E402
import candgen  # noqa: E402
import features as F  # noqa: E402
import model as M  # noqa: E402
import pairs as P  # noqa: E402

ROOT = M.ROOT
CACHE = f"{ROOT}/work/cache"
WORK = f"{ROOT}/work/wide"
KF = int(os.environ.get("KF", "100"))
KR = int(os.environ.get("KR", "20"))
BUDGET, DF_MAX = 40000, 30000
OUT = os.environ.get("OUT", "c2")
CHUNK = 10_000_000
SAMPLE_PER_PART = 2_000_000
CAP = 250

CHEAP_NAMES = (F.set_feature_names()
               + ["s", "fr", "rr", "s_rel_q", "s_rel_c", "q_npool", "c_npool",
                  "ad_eq", "adl_eq", "nm_eq", "q_ad_s1", "c_ad_s1", "c_ad_all",
                  "c_nm_s1", "q_nm_s1"])


# ------------------------------------------------------------------ pool
def wide_pool(split, country, source, idx1, ptr1, val1, n1):
    rowsT, ptrT, valT = block.load_terms(split, source, country)
    nT = len(rowsT)
    idxT = block.Index(ptrT, valT, nT)
    t0 = time.time()
    c, s, n, _ = block.query_index(idxT, ptr1, val1, df_max=DF_MAX,
                                   budget=BUDGET, topk=KF, max_touch=120000)
    qf, cf, sf, rf = candgen.pairs_from_topk(c, s, n, n1)
    del c, s, n, idxT
    c, s, n, _ = block.query_index(idx1, ptrT, valT, df_max=DF_MAX,
                                   budget=BUDGET, topk=KR, max_touch=120000)
    cr, qr, sr, rr = candgen.pairs_from_topk(c, s, n, nT)
    del c, s, n
    q, cc, sc, fr, rr2 = candgen.union_pairs(qf, cf, sf, rf, qr, cr, sr, rr,
                                             n1, nT)
    print(f"    pool {country} S{source}: {len(q):,} pairs ({len(q) / n1:.1f}/S1)"
          f"  {time.time() - t0:.0f}s", flush=True)
    return rowsT, nT, q, cc, sc, fr, rr2


@numba.njit(cache=True)
def _group_max(q, s, nq):
    mx = np.zeros(nq, np.float32)
    cnt = np.zeros(nq, np.int32)
    for i in range(q.shape[0]):
        if s[i] > mx[q[i]]:
            mx[q[i]] = s[i]
        cnt[q[i]] += 1
    return mx, cnt


def cheap_feats(qs, cs, q, c, s, fr, rr, qmax, cmax, qn, cn):
    n = len(q)
    X = np.empty((n, len(CHEAP_NAMES)), np.float32)
    qi, ci = q.astype(np.int64), c.astype(np.int64)
    for b, sh in enumerate(F.SHORT):
        F._set_feats(qi, ci, qs.ptr[sh], qs.val[sh], qs.idf[sh],
                     cs.ptr[sh], cs.val[sh], cs.idf[sh], X, b * F.N_SET)
    k = len(F.SHORT) * F.N_SET
    X[:, k + 0] = s
    X[:, k + 1] = np.minimum(fr, CAP)
    X[:, k + 2] = np.minimum(rr, CAP)
    X[:, k + 3] = s / np.maximum(qmax[qi], 1e-9)
    X[:, k + 4] = s / np.maximum(cmax[ci], 1e-9)
    X[:, k + 5] = qn[qi]
    X[:, k + 6] = cn[ci]
    for j, key in enumerate(("ad", "adl", "nm")):
        a, b = qs.sig[key][qi], cs.sig[key][ci]
        X[:, k + 7 + j] = (a == b) & (a != 0)
    X[:, k + 10] = qs.sig["ad_s1"][qi]
    X[:, k + 11] = cs.sig["ad_s1"][ci]
    X[:, k + 12] = np.log1p(cs.sig["ad_all"][ci])
    X[:, k + 13] = cs.sig["nm_s1"][ci]
    X[:, k + 14] = qs.sig["nm_s1"][qi]
    return X


class PoolPart:
    """Wide pool of one (country, source) with everything needed to featurize."""

    def __init__(self, split, country, source, S1, idft):
        idx1, ptr1, val1, rows1 = S1
        n1 = len(rows1)
        self.rows1 = rows1
        (self.rowsT, self.nT, self.q, self.c, self.s, self.fr,
         self.rr) = wide_pool(split, country, source, idx1, ptr1, val1, n1)
        self.qs = F.FieldStore(split, country, 1, idft)
        self.cs = F.FieldStore(split, country, source, idft)
        assert np.array_equal(self.qs.rows, rows1)
        assert np.array_equal(self.cs.rows, self.rowsT)
        self.qmax, self.qn = _group_max(self.q, self.s, n1)
        self.cmax, self.cn = _group_max(self.c, self.s, self.nT)
        self.source = source

    def feats(self, sel):
        return cheap_feats(self.qs, self.cs, self.q[sel], self.c[sel],
                           self.s[sel], self.fr[sel], self.rr[sel], self.qmax,
                           self.cmax, self.qn, self.cn)

    def labels(self, gt):
        s1row, pq, ps, tgt_row = gt
        qpos = np.full(int(max(s1row.max(), self.rows1.max())) + 2, -1, np.int64)
        qpos[self.rows1] = np.arange(len(self.rows1))
        locT = np.full(int(max(self.rowsT.max(), tgt_row.max())) + 2, -1, np.int64)
        locT[self.rowsT] = np.arange(self.nT)
        sel = (ps == self.source) & (qpos[s1row[pq]] >= 0)
        gq = qpos[s1row[pq[sel]]]
        gc = locT[tgt_row[sel]]
        ok = gc >= 0
        return P.gold_mask(self.q, self.c, self.nT, gq[ok], gc[ok]), gq[ok]


def s1_side(split, country):
    rows1, ptr1, val1 = block.load_terms(split, 1, country)
    return block.Index(ptr1, val1, len(rows1)), ptr1, val1, rows1


# ---------------------------------------------------------------- models
def train_cheap(X, y, f):
    import lightgbm as lgb
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=63,
               min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, max_bin=63, verbosity=-1,
               num_threads=int(os.environ.get("LGB_THREADS", "20")))
    models = {}
    for name, folds in (("A", (1, 2)), ("B", (3, 4))):
        t0 = time.time()
        tr = np.isin(f, folds)
        rng = np.random.default_rng(7)
        hold = rng.random(len(y)) < 0.05
        ds = lgb.Dataset(X[tr & ~hold], label=y[tr & ~hold],
                         feature_name=CHEAP_NAMES)
        vs = lgb.Dataset(X[tr & hold], label=y[tr & hold], reference=ds)
        b = lgb.train(prm, ds, num_boost_round=300, valid_sets=[vs],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
        b.save_model(f"{WORK}/cheap_{name}.txt")
        models[name] = b
        print(f"  cheap model {name} (folds {folds}): {b.num_trees()} trees "
              f"{time.time() - t0:.0f}s", flush=True)
    imp = sorted(zip(CHEAP_NAMES, models["A"].feature_importance("gain")),
                 key=lambda t: -t[1])
    print("  cheap top features:", ", ".join(n for n, _ in imp[:15]))
    return models


def load_cheap():
    import lightgbm as lgb
    return {k: lgb.Booster(model_file=f"{WORK}/cheap_{k}.txt") for k in "AB"}


def score_pool(pp, models, qfold):
    """Cross-fitted cheap probability for every pool pair."""
    n = len(pp.q)
    pc = np.empty(n, np.float32)
    pf = qfold[pp.q] if qfold is not None else np.zeros(n, np.int8)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        X = pp.feats(slice(a, b))
        f = pf[a:b]
        useA = np.isin(f, (3, 4))
        useB = np.isin(f, (1, 2))
        both = ~(useA | useB)
        out = np.zeros(b - a, np.float64)
        if useA.any():
            out[useA] = models["A"].predict(X[useA])
        if useB.any():
            out[useB] = models["B"].predict(X[useB])
        if both.any():
            out[both] = 0.5 * (models["A"].predict(X[both])
                               + models["B"].predict(X[both]))
        pc[a:b] = out
        del X
    return pc


def group_rank(key, val):
    o = np.lexsort((-val, key))
    k = key[o]
    first = np.ones(len(k), bool)
    first[1:] = k[1:] != k[:-1]
    starts = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    r = np.empty(len(k), np.int32)
    r[o] = (np.arange(len(k)) - starts[gid]).astype(np.int32)
    return r


# ------------------------------------------------------------------ main
RULES = [("prod fwd<20|rev<5", None, None, True),
         ("prod fwd<12|rev<5", None, None, "p12")]
for M_ in (5, 8, 10, 15, 20, 30):
    for R_ in (0, 2, 3):
        RULES.append((f"cheap q<{M_} c<{R_}", M_, R_, False))
for M_ in (3, 5, 10):
    RULES.append((f"prod + cheap q<{M_}", M_, 0, True))


def rule_mask(r, fr, rr, rq, rc):
    name, m_, r_, prod = r
    if prod == "p12":
        return (fr < 12) | (rr < 5)
    keep = np.zeros(len(fr), bool)
    if prod:
        keep |= (fr < 20) | (rr < 5)
    if m_:
        keep |= rq < m_
    if r_:
        keep |= rc < r_
    return keep


def run(split):
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(f"{CACHE}/{split}/cand", exist_ok=True)
    t00 = time.time()
    meta1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet", columns=["id"])
    qfold_all = M.fold_of(meta1["id"].to_numpy()) if split == "train" else None
    gt = M.gt_arrays(split) if split == "train" else None
    countries = M.countries_of(split)

    if split == "train" and not os.path.exists(f"{WORK}/cheap_B.txt"):
        Xs, ys, fs = [], [], []
        for country in countries:
            idft = M.idf_tables(split, country)
            S1 = s1_side(split, country)
            for source in (2, 3):
                pp = PoolPart(split, country, source, S1, idft)
                lab, _ = pp.labels(gt)
                qf = qfold_all[pp.rows1][pp.q]
                cand = np.flatnonzero(qf != 0)
                rng = np.random.default_rng(11 + source)
                sel = np.sort(rng.choice(cand, min(SAMPLE_PER_PART, len(cand)),
                                         replace=False))
                Xs.append(pp.feats(sel)); ys.append(lab[sel]); fs.append(qf[sel])
                print(f"    sampled {len(sel):,} ({lab[sel].mean() * 100:.2f}% pos)",
                      flush=True)
                del pp; gc.collect()
            del S1; gc.collect()
        X = np.concatenate(Xs); y = np.concatenate(ys); f = np.concatenate(fs)
        del Xs, ys, fs
        models = train_cheap(X, y, f)
        del X, y, f; gc.collect()
    else:
        models = load_cheap()

    stats = {r[0]: [0, 0] for r in RULES}
    n_val_s1 = n_val_true = n_val_pool = 0
    for country in countries:
        idft = M.idf_tables(split, country)
        S1 = s1_side(split, country)
        for source in (2, 3):
            t0 = time.time()
            pp = PoolPart(split, country, source, S1, idft)
            qfold = qfold_all[pp.rows1] if qfold_all is not None else None
            pc = score_pool(pp, models, qfold)
            rq = group_rank(pp.q.astype(np.int64), pc)
            rc = group_rank(pp.c.astype(np.int64), pc)
            if split == "train":
                lab, gq = pp.labels(gt)
                isv = qfold[pp.q] == 0
                nv = int((qfold == 0).sum())
                if source == 2:
                    n_val_s1 += nv
                n_val_true += int((qfold[gq] == 0).sum())
                n_val_pool += int((lab & isv).sum())
                for r in RULES:
                    k = rule_mask(r, pp.fr, pp.rr, rq, rc) & isv
                    stats[r[0]][0] += int(k.sum())
                    stats[r[0]][1] += int((k & lab).sum())
            keep = (pp.fr < 20) | (pp.rr < 5) | (rq < 40) | (rc < 5)
            np.savez(f"{CACHE}/{split}/cand/{OUT}all_{country}_s{source}.npz",
                     rows1=pp.rows1, rowsT=pp.rowsT, q=pp.q[keep], c=pp.c[keep],
                     s=pp.s[keep], frank=pp.fr[keep], rrank=pp.rr[keep],
                     pc=pc[keep], rq=rq[keep].astype(np.int16),
                     rc=rc[keep].astype(np.int16))
            print(f"  {split} {country} S{source}: pool {len(pp.q):,} -> saved "
                  f"{keep.sum():,} ({keep.sum() / len(pp.rows1):.1f}/S1)  "
                  f"{time.time() - t0:.0f}s", flush=True)
            del pp, pc, rq, rc; gc.collect()
        del S1; gc.collect()

    if split == "train":
        print(f"\nval fold: {n_val_s1:,} S1, {n_val_true:,} true pairs, wide pool "
              f"recall {n_val_pool / n_val_true * 100:.3f}%")
        print(f"  {'rule':<22} {'cand/S1':>8} {'pair recall':>12}")
        for name, (nc, nt) in stats.items():
            print(f"  {name:<22} {nc / n_val_s1:>8.2f} {nt / n_val_true * 100:>11.3f}%")
    print(f"total {time.time() - t00:.0f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "train")
