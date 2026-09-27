"""
Two-stage scorer: cross-fitted pair model (stage 1) + a stacked model that
sees the *local graph* around every pair (stage 2).

Why a second stage.  Stage 1 judges each (S1, candidate) pair in isolation;
its only view of the competition is the *retrieval* score of rivals.  But the
decisions that still go wrong are relational:

  * best-vs-second:  a 0.8 candidate is safe when the runner-up is 0.1 and a
    coin flip when it is 0.79 -- needs the *model* scores of siblings;
  * many-to-one:     a target claimed at 0.95 by another S1 entity is almost
    surely not ours, one nobody else wants is more likely ours;
  * transitivity:    a candidate whose name was replaced ("Raven LLC" ->
    "Lumrizagild") still shares its address with the entity's *other*
    confidently-matched records.

Stage 2 gets exactly those signals, computed from out-of-fold stage-1
probabilities so it never sees a score produced by a model trained on the
same entity.

Cross-fitting.  Entity folds 0..4 (fold 0 = validation, never trained on).
Model k (k=1..4) trains on folds {1..4} minus k.  Pair p1 = model_k for
fold-k entities; validation and test pairs get the mean of all four.

Usage:
    python stack.py train            # stage-1 x4, stage-2, validation npz
Env: TAG (default s1), CTAG (candidate tag, c1), KEEP_P, MAX_PER_PART.
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
import model as M  # noqa: E402
import pairs as P  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
MODELS = f"{ROOT}/work/models"
TAG = os.environ.get("TAG", "s1")
WORK = f"{ROOT}/work/stack/{TAG}"
CTAG = os.environ.get("CTAG", "c1")
VAL_FOLD = 0
FOLDS = [1, 2, 3, 4]
KEEP_P = float(os.environ.get("KEEP_P", "0.003"))
MAX_PER_PART = int(os.environ.get("MAX_PER_PART", "2000000"))
NROUND = int(os.environ.get("NROUND", "600"))
CHUNK = 2_000_000
# per-country experts: restrict both stages to these countries (default: all)
COUNTRIES = [c for c in os.environ.get("COUNTRIES", "").split(",") if c]
# leave-one-country-out: countries scored for validation only, never trained on
EVAL_COUNTRIES = [c for c in os.environ.get("EVAL_COUNTRIES", "").split(",") if c]
# self-training, "Country|split|dump.npz|hi|lo[;...]": an unlabelled country's
# pairs, pseudo-labelled from an earlier run's p2 dump (infer2.py DUMP=),
# join stage-1 training of every fold model.  pos = p2>=hi and the target's
# most confident claimant; neg = p2<lo (pairs absent from the dump had p2<0.02)
PSEUDO = [x for x in os.environ.get("PSEUDO", "").split(";") if x]
PSEUDO_FOLD = 9   # never equals a cross-fit fold, so every stage-1 model sees it
# "explained": a low-p pair is a negative only if its S1 already has a confident
# match or its target is confidently claimed by another S1 -- the unexplained
# low-p pairs are where the first model's missed true matches hide
PSEUDO_NEG = os.environ.get("PSEUDO_NEG", "all")


def countries(split):
    return [c for c in M.countries_of(split) if not COUNTRIES or c in COUNTRIES]


# ----------------------------------------------------------------- stage 1
def build_stage1_matrix(split="train"):
    s1row, pq, ps, tgt_row = M.gt_arrays(split)
    ids1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                           columns=["id"])["id"].to_numpy()
    fold = M.fold_of(ids1)
    nf = len(F.feature_names())
    parts = [(c, s) for c in countries(split) for s in (2, 3)]
    nrows = (len(parts) + 2 * len(PSEUDO)) * MAX_PER_PART
    X = np.empty((nrows, nf), np.float32)
    y = np.empty(nrows, np.int8)
    f = np.empty(nrows, np.int8)
    filled = 0
    for country, source in parts:
        t0 = time.time()
        idft = M.idf_tables(split, country)
        part = M.Partition(split, country, source, CTAG, idft)
        lab, _ = part.labels(s1row, pq, ps, tgt_row)
        qfold = fold[part.rows1][part.q]
        sel = (qfold != VAL_FOLD).nonzero()[0]
        if len(sel) > MAX_PER_PART:
            rng = np.random.default_rng(17)
            sel = np.sort(rng.choice(sel, MAX_PER_PART, replace=False))
        for a in range(0, len(sel), CHUNK):
            sub = sel[a:a + CHUNK]
            X[filled + a:filled + a + len(sub)] = F.compute(
                part.qs, part.cs, part.q[sub], part.c[sub], part.ctx[sub])
        y[filled:filled + len(sel)] = lab[sel]
        f[filled:filled + len(sel)] = qfold[sel]
        filled += len(sel)
        print(f"  s1-train {country:7s} S{source}: {len(sel):>9,} pairs "
              f"({lab[sel].mean() * 100:5.2f}% pos) {time.time() - t0:.0f}s",
              flush=True)
        part.free(); del part, lab
        F._HMEMO.clear(); gc.collect()
    for spec in PSEUDO:
        filled = add_pseudo(spec, X, y, f, filled)
    return X[:filled], y[:filled], f[:filled]


def add_pseudo(spec, X, y, f, filled):
    """Append one country's pseudo-labelled pairs; returns the new fill level."""
    country, psplit, path, hi, lo = spec.split("|")
    hi, lo = float(hi), float(lo)
    z = np.load(path)
    dsrc, drow, dtid, dp = z["src"], z["row"].astype(np.int64), \
        z["tid"].astype(np.int64), z["p"].astype(np.float64)
    big = int(max(dtid.max(), 1)) + 2
    # target exclusivity inside the dump: only the best claimant may be positive
    tkey = dsrc.astype(np.int64) * big + dtid
    o = np.lexsort((-dp, tkey))
    win = np.zeros(len(dp), bool)
    win[o[np.concatenate([[True], tkey[o][1:] != tkey[o][:-1]])]] = True
    conf = win & (dp >= hi)
    pos_rows = np.unique(drow[conf])
    claimed = np.unique(tkey[conf])
    audit = M.gt_arrays(psplit) if psplit == "train" else None
    idft = M.idf_tables(psplit, country)
    for source in (2, 3):
        t0 = time.time()
        part = M.Partition(psplit, country, source, CTAG, idft)
        tids = pl.read_parquet(f"{CACHE}/{psplit}/s{source}/meta.parquet",
                               columns=["id"])["id"].to_numpy()[part.rowsT]
        key = part.rows1[part.q].astype(np.int64) * big + tids[part.c].astype(np.int64)
        m = dsrc == source
        dk = drow[m] * big + dtid[m]
        od = np.argsort(dk)
        dk, pd_, wd = dk[od], dp[m][od], win[m][od]
        j = np.clip(np.searchsorted(dk, key), 0, max(len(dk) - 1, 0))
        hit = (dk[j] == key) if len(dk) else np.zeros(len(key), bool)
        p = np.where(hit, pd_[j], 0.0)
        pos = hit & (p >= hi) & wd[j]
        neg = p < lo
        if PSEUDO_NEG == "explained":
            neg &= (np.isin(part.rows1[part.q], pos_rows)
                    | np.isin(source * big + tids[part.c].astype(np.int64), claimed))
        sel = (pos | neg).nonzero()[0]
        rng = np.random.default_rng(41 + source)
        if len(sel) > MAX_PER_PART:
            sel = np.sort(rng.choice(sel, MAX_PER_PART, replace=False))
        for a in range(0, len(sel), CHUNK):
            sub = sel[a:a + CHUNK]
            X[filled + a:filled + a + len(sub)] = F.compute(
                part.qs, part.cs, part.q[sub], part.c[sub], part.ctx[sub])
        yl = pos[sel].astype(np.int8)
        y[filled:filled + len(sel)] = yl
        f[filled:filled + len(sel)] = PSEUDO_FOLD
        filled += len(sel)
        msg = ""
        if audit is not None:
            lab, _ = part.labels(*audit)
            tl = lab[sel].astype(bool)
            msg = (f"  audit: pseudo-pos precision {tl[yl == 1].mean():.4f}, "
                   f"pseudo-neg purity {1 - tl[yl == 0].mean():.5f}, "
                   f"true pos covered {tl[yl == 1].sum() / max(lab.sum(), 1):.3f}")
        print(f"  pseudo {psplit}/{country} S{source}: {len(sel):>9,} pairs "
              f"({yl.mean() * 100:5.2f}% pos){msg} {time.time() - t0:.0f}s", flush=True)
        part.free(); del part
        F._HMEMO.clear(); gc.collect()
    return filled


def train_stage1(X, y, f):
    import lightgbm as lgb
    models = []
    for k in FOLDS:
        t0 = time.time()
        tr = (f != k).nonzero()[0]
        rng = np.random.default_rng(100 + k)
        hold = rng.random(len(tr)) < 0.08
        b = M.train_lgbm(X[tr[~hold]], y[tr[~hold]], num_round=NROUND,
                         valid=(X[tr[hold]], y[tr[hold]]))
        b.save_model(f"{WORK}/stage1_f{k}.txt")
        models.append(b)
        print(f"  stage-1 model {k}: {b.num_trees()} trees "
              f"{time.time() - t0:.0f}s", flush=True)
    return models


def load_stage1():
    import lightgbm as lgb
    return [lgb.Booster(model_file=f"{WORK}/stage1_f{k}.txt") for k in FOLDS]


def predict_p1(models, X, rfold):
    """OOF p1: fold-k rows use model k; other rows (val/test) the mean."""
    p = np.zeros(len(X), np.float64)
    avg = ~np.isin(rfold, FOLDS)
    for k, b in zip(FOLDS, models):
        m = avg | (rfold == k)
        if not m.any():
            continue
        pk = b.predict(X[m], num_iteration=b.best_iteration or None)
        w = np.where(avg[m], 1.0 / len(models), 1.0)
        p[m] += w * pk
    return p.astype(np.float32)


# ------------------------------------------------------ scoring one country
def score_country(split, country, models, fold1=None, labels=None):
    """Stage-1 scores for every candidate pair of one country (S2 and S3).

    Returns a dict of flat per-pair arrays plus base features of the pairs
    with p1 >= KEEP_P (the only pairs stage 2 will ever look at).
    fold1: per global S1 row fold id (None -> all rows averaged, i.e. test).
    labels: (s1row, pq, ps, tgt_row) on train, None on test.
    """
    idft = M.idf_tables(split, country)
    out = {k: [] for k in ("q", "c", "src", "p1", "y", "rf", "keep",
                           "rows1", "Xk", "fr", "rr")}
    stores = {}
    for source in (2, 3):
        t0 = time.time()
        part = M.Partition(split, country, source, CTAG, idft)
        n = len(part.q)
        rfold = (fold1[part.rows1][part.q] if fold1 is not None
                 else np.full(n, VAL_FOLD, np.int8))
        if labels is not None:
            lab, _ = part.labels(*labels)
        else:
            lab = np.zeros(n, bool)
        p1 = np.empty(n, np.float32)
        Xk = []
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            X = F.compute(part.qs, part.cs, part.q[a:b], part.c[a:b],
                          part.ctx[a:b])
            p1[a:b] = predict_p1(models, X, rfold[a:b])
            k = p1[a:b] >= KEEP_P
            Xk.append(X[k])
            del X
        keep = p1 >= KEEP_P
        out["q"].append(part.q); out["c"].append(part.c)
        out["src"].append(np.full(n, source, np.int8)); out["p1"].append(p1)
        out["y"].append(lab.astype(bool)); out["rf"].append(rfold)
        out["keep"].append(keep); out["Xk"].append(np.concatenate(Xk))
        out["fr"].append(part.frank); out["rr"].append(part.rrank)
        out["rows1"] = part.rows1
        stores[source] = (part.rowsT, part.cs)
        stores['q'] = part.qs
        print(f"  scored {split}/{country} S{source}: {n:,} pairs, "
              f"{keep.sum():,} kept (p1>={KEEP_P}), "
              f"pos kept {lab[keep].sum():,}/{lab.sum():,}  "
              f"{time.time() - t0:.0f}s", flush=True)
        del part.qs
        del part
        F._HMEMO.clear(); gc.collect()
    D = {k: (np.concatenate(v) if k not in ("rows1",) else v)
         for k, v in out.items()}
    return D, stores


# ------------------------------------------------------- stage-2 features
S2_GROUP = ["p1", "lg_p1", "g_rank", "g_max", "g_second", "g_rel", "g_margin",
            "g_gap_next", "g_sum", "g_n50", "g_n90", "g_n",
            "t_rank", "t_best_rival", "t_n", "t_n50", "t_margin",
            "a1_p", "a2_p", "a1_same_src"]
ANCHOR_FIELDS = ["at", "nt", "nm", "as"]


S2X = os.environ.get("S2X", "0") == "1"
S2X_NAMES = ["x_nm_has", "x_nm_absdiff", "x_nm_reldiff", "x_nm_prefix", "x_nm_suffix",
             "x_nm_onedigit", "x_nm_lendiff", "x_alt_sup", "x_num_in_s1", "x_ent_same",
             "x_seg", "x_n_altclust"]


def _digits(x):
    return np.floor(np.log10(np.maximum(x, 1))).astype(np.int64) + 1


def num_relation(a, b):
    """Typo vs neighbouring building, from the two records' largest numbers."""
    has = (a >= 0) & (b >= 0)
    a = np.maximum(a, 0); b = np.maximum(b, 0)
    d = np.abs(a - b)
    rel = d / np.maximum(np.maximum(a, b), 1)
    la, lb = _digits(a), _digits(b)
    big, small = np.maximum(a, b), np.minimum(a, b)
    lbig, lsm = np.maximum(la, lb), np.minimum(la, lb)
    longer = has & (lbig > lsm)
    prefix = longer & (big // (10 ** np.clip(lbig - lsm, 0, 18)) == small)
    suffix = longer & (big % (10 ** np.clip(lsm, 0, 18)) == small)
    x, y = a.copy(), b.copy()
    nd = np.zeros(len(a), np.int64)
    for _ in range(19):
        nd += ((x % 10) != (y % 10)) & ((x > 0) | (y > 0))
        x //= 10; y //= 10
    one = has & (la == lb) & (nd == 1)
    return np.stack([has, np.where(has, d, -1), np.where(has, rel, -1), prefix,
                     suffix, one, np.where(has, la - lb, 0)], 1).astype(np.float32)


def _explode(ptr, val, idx):
    lens = (ptr[idx + 1] - ptr[idx]).astype(np.int64)
    rep = np.repeat(np.arange(len(idx)), lens)
    start = np.repeat(ptr[idx], lens)
    off = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    return rep, val[start + off].astype(np.int64)


def cluster_feats(q, c, src, qs, cstores):
    """Rival-cluster structure of each entity's candidate set (see shiftfix.py)."""
    n = len(q)
    q = q.astype(np.int64); c = c.astype(np.int64)
    reps, hs = [], []
    for s, st in cstores.items():
        m = np.flatnonzero(src == s)
        r, h = _explode(st.ptr["nm"], st.val["nm"], c[m])
        reps.append(m[r]); hs.append(h)
    pi = np.concatenate(reps); h = np.concatenate(hs)
    qr, qh = _explode(qs.ptr["nm"], qs.val["nm"], np.arange(qs.n))
    qkey = np.sort(qr.astype(np.int64) * (1 << 32) + qh)
    k = q[pi] * (1 << 32) + h
    if len(qkey):
        j = np.clip(np.searchsorted(qkey, k), 0, len(qkey) - 1)
        inq = qkey[j] == k
    else:
        inq = np.zeros(len(k), bool)
    uk, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    sup = cnt[inv] - 1
    alt = np.zeros(n, np.int32); np.maximum.at(alt, pi[~inq], sup[~inq])
    hasnum = np.zeros(n, bool); hasnum[pi] = True
    anyin = np.zeros(n, bool); anyin[pi[inq]] = True
    ent_same = np.zeros(qs.n, bool); ent_same[q[anyin]] = True
    seg = np.full(n, 3, np.int8); seg[~hasnum] = 4; seg[anyin] = 0
    rest = hasnum & ~anyin & (alt >= 1)
    seg[rest & ent_same[q]] = 1; seg[rest & ~ent_same[q]] = 2
    # distinct rival numbers (not the S1's) held by >=2 candidates, per entity
    uinq = np.zeros(len(uk), bool); uinq[inv[inq]] = True
    rival = (cnt >= 2) & ~uinq
    nclu = np.bincount((uk[rival] >> 32).astype(np.int64), minlength=qs.n)
    return np.stack([alt, anyin, ent_same[q], seg, nclu[q]], 1).astype(np.float32)


def s2_feature_names():
    names = [f"b_{x}" for x in F.feature_names()] + S2_GROUP
    for sh in ANCHOR_FIELDS:
        names += [f"a1_{sh}_{x}" for x in ("inter", "jac", "cont", "widf",
                                           "wjac", "mxsh", "mxq", "mxc", "minn")]
    names += ["a2_at_wjac", "a2_nt_wjac", "a2_at_cont"]
    if S2X:
        names += S2X_NAMES
    return names


def _logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def s2_features(D, stores):
    """Stage-2 matrix for the kept pairs of one country (row order = kept)."""
    q, c, src, p1 = D["q"], D["c"], D["src"], D["p1"].astype(np.float64)
    n = len(q)
    kept = D["keep"].nonzero()[0]
    nk = len(kept)
    G = np.zeros((n, len(S2_GROUP)), np.float32)
    # ---- S1 group (the entity's whole candidate list, both sources)
    rank, gmax, second, size = P.group_stats(q, p1)
    nq = int(q.max()) + 1
    gsum = np.bincount(q, weights=p1, minlength=nq)
    g50 = np.bincount(q, weights=(p1 > 0.5), minlength=nq)
    g90 = np.bincount(q, weights=(p1 > 0.9), minlength=nq)
    o = np.lexsort((-p1, q))
    qs_, ps_ = q[o], p1[o]
    same_next = np.zeros(n, bool); same_next[:-1] = qs_[1:] == qs_[:-1]
    nxt = np.where(same_next, np.append(ps_[1:], 0.0), 0.0)
    gap_next = np.empty(n); gap_next[o] = ps_ - nxt
    # anchors: best two *other* candidates of the same entity
    first = np.ones(n, bool); first[1:] = qs_[1:] != qs_[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    pos_in = np.arange(n) - start
    gsz = size[o]
    a1 = np.where(pos_in == 0, start + 1, start)
    a2 = np.where(pos_in <= 1, start + 2, start + 1)
    a1 = np.where(a1 - start < gsz, a1, -1)
    a2 = np.where(a2 - start < gsz, a2, -1)
    A1 = np.full(n, -1, np.int64); A1[o] = np.where(a1 >= 0, o[np.maximum(a1, 0)], -1)
    A2 = np.full(n, -1, np.int64); A2[o] = np.where(a2 >= 0, o[np.maximum(a2, 0)], -1)
    G[:, 0] = p1
    G[:, 1] = _logit(p1)
    G[:, 2] = rank
    G[:, 3] = gmax
    G[:, 4] = second
    G[:, 5] = p1 / np.maximum(gmax, 1e-9)
    G[:, 6] = np.where(rank == 0, p1 - second, p1 - gmax)
    G[:, 7] = gap_next
    G[:, 8] = gsum[q]
    G[:, 9] = g50[q]
    G[:, 10] = g90[q]
    G[:, 11] = size
    # ---- target competition (many-to-one): every S1 claiming this record
    key = src.astype(np.int64) * (int(c.max()) + 2) + c
    trank, tmax, tsecond, tsize = P.group_stats(key, p1)
    tcnt50 = np.bincount(np.unique(key, return_inverse=True)[1],
                         weights=(p1 > 0.5))
    _, tinv = np.unique(key, return_inverse=True)
    rival = np.where(trank == 0, tsecond, tmax)
    G[:, 12] = trank
    G[:, 13] = rival
    G[:, 14] = tsize
    G[:, 15] = tcnt50[tinv]
    G[:, 16] = p1 - rival
    G[:, 17] = np.where(A1 >= 0, p1[np.maximum(A1, 0)], -1)
    G[:, 18] = np.where(A2 >= 0, p1[np.maximum(A2, 0)], -1)
    G[:, 19] = np.where(A1 >= 0, src[np.maximum(A1, 0)] == src, -1)

    # ---- transitive: candidate vs its entity's best other candidates
    nA = len(ANCHOR_FIELDS) * F.N_SET
    AF = np.full((nk, nA + 3), -2.0, np.float32)
    for which, Aidx, cols in ((1, A1, None), (2, A2, None)):
        ak = Aidx[kept]
        for sa in (2, 3):
            for sc in (2, 3):
                m = (ak >= 0) & (src[np.maximum(ak, 0)] == sa) & (src[kept] == sc)
                if not m.any():
                    continue
                ai = c[ak[m]].astype(np.int64)
                ci = c[kept[m]].astype(np.int64)
                _, st_a = stores[sa]
                _, st_c = stores[sc]
                tmp = np.zeros((m.sum(), nA), np.float32)
                for b, sh in enumerate(ANCHOR_FIELDS):
                    F._set_feats(ai, ci, st_a.ptr[sh], st_a.val[sh], st_a.idf[sh],
                                 st_c.ptr[sh], st_c.val[sh], st_c.idf[sh],
                                 tmp, b * F.N_SET)
                if which == 1:
                    AF[m, :nA] = tmp
                else:
                    AF[m, nA + 0] = tmp[:, 0 * F.N_SET + 4]   # at wjac
                    AF[m, nA + 1] = tmp[:, 1 * F.N_SET + 4]   # nt wjac
                    AF[m, nA + 2] = tmp[:, 0 * F.N_SET + 2]   # at cont
    if not S2X:
        return np.hstack([D["Xk"], G[kept], AF])
    qs = stores["q"]
    cst = {sv: stores[sv][1] for sv in (2, 3)}
    a = qs.num_max[q[kept]]
    b = np.empty(nk, np.int64)
    for sv in (2, 3):
        m = src[kept] == sv
        b[m] = cst[sv].num_max[c[kept][m]]
    XN = num_relation(a, b)
    XC = cluster_feats(q, c, src, qs, cst)[kept]
    return np.hstack([D["Xk"], G[kept], AF, XN, XC])


# -------------------------------------------------------------- training
def train_stage2(X, y, idx, hold_seed=5):
    """Train on rows `idx` of X.  LightGBM subsets one binned copy of X instead
    of slicing X twice: a US-sized stage 2 otherwise peaks near 16 GB."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(hold_seed)
    hold = rng.random(len(idx)) < 0.10
    env = os.environ.get
    p = dict(objective="binary", learning_rate=0.05,
             num_leaves=int(env("S2_LEAVES", "127")),
             min_data_in_leaf=int(env("S2_MINLEAF", "200")),
             feature_fraction=float(env("S2_FF", "0.7")), bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=float(env("S2_L2", "1.0")), max_bin=127,
             verbosity=-1, num_threads=int(env("LGB_THREADS", "20")),
             metric=["binary_logloss"])
    names = s2_feature_names()
    if env("S2_MONO", "0") == "1":
        mono = M.mono_vector(names, prefix="b_")
        for k in ("p1", "lg_p1"):
            mono[names.index(k)] = 1
        p.update(monotone_constraints=mono,
                 monotone_constraints_method="intermediate")
    full = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=False,
                       params=dict(max_bin=127, verbosity=-1))
    ds = full.subset(idx[~hold])
    vs = full.subset(idx[hold])
    b = lgb.train(p, ds, num_boost_round=int(env("S2_NROUND", "1500")),
                  valid_sets=[vs],
                  callbacks=[lgb.log_evaluation(100),
                             lgb.early_stopping(50, verbose=True)])
    raw = b.predict(X[idx[hold]], num_iteration=b.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y[idx[hold]])
    return b, iso


def main_train():
    os.makedirs(WORK, exist_ok=True)
    t00 = time.time()
    split = "train"
    labels = M.gt_arrays(split)
    s1row, pq, ps, tgt_row = labels
    meta1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet",
                            columns=["id", "country"])
    ids1 = meta1["id"].to_numpy()
    fold1 = M.fold_of(ids1)
    n_gt = len(s1row)
    g2gt = np.full(len(ids1), -1, np.int64); g2gt[s1row] = np.arange(n_gt)
    T = np.zeros(n_gt, np.int32); np.add.at(T, pq, 1)

    # ---- stage 1
    mpath = f"{WORK}/stage1_f{FOLDS[-1]}.txt"
    if os.path.exists(mpath):
        models = load_stage1()
        print("stage-1 models loaded", flush=True)
    else:
        X, y, f = build_stage1_matrix(split)
        print(f"stage-1 matrix {X.shape} {X.nbytes / 1e9:.1f} GB "
              f"pos {y.mean() * 100:.2f}%", flush=True)
        models = train_stage1(X, y, f)
        imp = sorted(zip(F.feature_names(),
                         models[0].feature_importance("gain")), key=lambda t: -t[1])
        print("stage-1 top features:", ", ".join(n for n, _ in imp[:20]))
        del X, y, f; gc.collect()

    # ---- score everything, build stage-2 rows
    # stage-2 rows spill to disk per country and are loaded into one
    # preallocated matrix: np.concatenate would hold two copies at once
    spill, S2y, S2f, VAL = [], [], [], {k: [] for k in
                                        ("gt", "p", "p2i", "pos", "src", "tid")}
    n2 = 0
    trained = countries(split)
    for country in trained + [c for c in EVAL_COUNTRIES if c not in trained]:
        D, stores = score_country(split, country, models, fold1, labels)
        Xs2 = s2_features(D, stores)
        kept = D["keep"].nonzero()[0]
        if country not in trained:   # eval-only: its val-fold rows are all we need
            m = D["rf"][kept] == VAL_FOLD
            Xs2, kept = Xs2[m], kept[m]
        rows1 = D["rows1"]
        path = f"{WORK}/_s2rows_{country}.npy"
        np.save(path, Xs2)
        spill.append((path, len(Xs2), Xs2.shape[1]))
        S2y.append(D["y"][kept].astype(np.int8))
        S2f.append(D["rf"][kept])
        # validation bookkeeping: every val pair, kept or not
        isval = D["rf"] == VAL_FOLD
        tids = {s: pl.read_parquet(f"{CACHE}/{split}/s{s}/meta.parquet",
                                   columns=["id"])["id"].to_numpy()[stores[s][0]]
                for s in (2, 3)}
        tid = np.where(D["src"] == 2, tids[2][np.minimum(D["c"], len(tids[2]) - 1)],
                       tids[3][np.minimum(D["c"], len(tids[3]) - 1)])
        kidx = np.full(len(D["q"]), -1, np.int64)
        kidx[kept] = np.arange(len(kept)) + n2
        n2 += len(kept)
        VAL["gt"].append(g2gt[rows1][D["q"][isval]])
        VAL["p"].append(D["p1"][isval])
        VAL["p2i"].append(kidx[isval])
        VAL["pos"].append(D["y"][isval])
        VAL["src"].append(D["src"][isval])
        VAL["tid"].append(tid[isval].astype(np.int32))
        del D, stores, Xs2; gc.collect()

    X2 = np.empty((n2, spill[0][2]), np.float32)
    off = 0
    for path, k, _ in spill:
        X2[off:off + k] = np.load(path, mmap_mode="r")
        off += k
        os.remove(path)
    y2 = np.concatenate(S2y); f2 = np.concatenate(S2f)
    del S2y, S2f; gc.collect()
    print(f"\nstage-2 rows {X2.shape}  pos {y2.mean() * 100:.2f}%  "
          f"{X2.nbytes / 1e9:.1f} GB", flush=True)
    tr = f2 != VAL_FOLD
    t0 = time.time()
    b2, iso2 = train_stage2(X2, y2, tr.nonzero()[0])
    b2.save_model(f"{WORK}/stage2.txt")
    with open(f"{WORK}/iso2.pkl", "wb") as fh:
        pickle.dump(iso2, fh)
    print(f"stage-2 trained: {b2.num_trees()} trees {time.time() - t0:.0f}s")
    imp = sorted(zip(s2_feature_names(), b2.feature_importance("gain")),
                 key=lambda t: -t[1])
    print("stage-2 top features:", ", ".join(f"{n}" for n, _ in imp[:25]))

    vi = (~tr).nonzero()[0]
    p2k = np.zeros(len(X2), np.float32)
    p2k[vi] = iso2.predict(b2.predict(X2[vi], num_iteration=b2.best_iteration))
    p2i = np.concatenate(VAL["p2i"])
    p2 = np.where(p2i >= 0, p2k[np.maximum(p2i, 0)], 0.0).astype(np.float32)
    np.savez(f"{MODELS}/val_scores_{TAG}.npz",
             gt=np.concatenate(VAL["gt"]), p=np.concatenate(VAL["p"]), p2=p2,
             pos=np.concatenate(VAL["pos"]), src=np.concatenate(VAL["src"]),
             tid=np.concatenate(VAL["tid"]), T=T, fold=fold1, g2gt=g2gt)
    print(f"saved val scores -> val_scores_{TAG}.npz  total {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main_train()
