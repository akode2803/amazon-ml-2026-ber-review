"""
Two-pipeline blend: our stack (+ cross-encoder) and the teammate's single-stage
c5 model (LB 0.964172; ours v5b 0.96101).  Both validate on the same entity
fold (identical fold_of), so a blend fitted on it is honest for both.

Pairs are joined on stable ids (S1 id, source, target id): the two pipelines
build their caches separately, so row numbers are not comparable.  Features
per pair: logit of each model's probability (floored; test dumps only carry
p >= 0.02, so val uses the same floor) and whether the pair was a candidate
of each pipeline.  Logistic regression per country; decision rule grid on
val; test dumps written in the infer2.py layout for assemble.py.

Env: OURS "US:s2xUS_P2ce:us_p2ce,India:s2xIN_P2ce:in_p2ce,France:s2xP3ce:fr_p3ce"
     (country : our val tag : our test dump), THEIRS_ROOT, THEIRS_TAG (c5),
     THEIRS_DUMP (c5_test).  France has no labels: its blend is fitted on the
     US+India val pairs of France's own (joint) model.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import waterfall as W  # noqa: E402

ROOT = W.ROOT
TROOT = os.environ.get("THEIRS_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ath")
TTAG = os.environ.get("THEIRS_TAG", "c5")
TDUMP = os.environ.get("THEIRS_DUMP", "c5_test")
DUMPD = f"{ROOT}/work/dumps"
OURS = [x.split(":") for x in os.environ.get(
    "OURS", "US:s2xUS_P2ce:us_p2ce,India:s2xIN_P2ce:in_p2ce,France:s2xP3ce:fr_p3ce").split(",")]
FLOOR = 0.02
GRID = [(t, a) for t in (1.0, 1.1, 1.2, 1.3) for a in (1.0, 2.0, 4.0)]
TIDMASK = (1 << 30) - 1


def s1_ids(root, split):
    m = pl.read_parquet(f"{root}/work/cache/{split}/s1/meta.parquet",
                        columns=["id", "country"])
    return m["id"].to_numpy().astype(np.int64), m["country"].to_numpy()


def pkey(s1id, src, tid):
    return (s1id.astype(np.int64) << 32) | (src.astype(np.int64) << 30) | tid.astype(np.int64)


def val_pairs(root, tag, key_p, countries):
    """Val-fold pairs of `countries` from a val npz, keyed by stable ids."""
    z = np.load(f"{root}/work/models/val_scores_{tag}.npz")
    s1row = np.load(f"{root}/work/cache/gt_train.npz")["s1row"]
    ids, ctry = s1_ids(root, "train")
    r1 = s1row[z["gt"]]
    m = (z["fold"][r1] == W.VAL_FOLD) & np.isin(ctry[r1], countries)
    k = pkey(ids[r1[m]], z["src"][m], z["tid"][m])
    o = np.argsort(k)
    return (k[o], z[key_p][m][o].astype(np.float64), z["pos"][m][o].astype(bool),
            ids[r1[m]][o])


def _lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def feats(po, ino, pt, int_):
    lo = _lg(np.where(po >= FLOOR, po, 0.005))
    lt = _lg(np.where(pt >= FLOOR, pt, 0.005))
    return np.stack([lo, lt, ino, int_, lo * lt / 10], 1)


def align(u, k):
    i = np.minimum(np.searchsorted(k, u), max(len(k) - 1, 0))
    return i, (k[i] == u) if len(k) else np.zeros(len(u), bool)


def macro_grid(g, p, y, src, tid, T):
    ex = W.exclusivity(p, src, tid)
    out = {}
    for ts, ab in GRID:
        tk, _ = W.decide_pairs(g, p, len(T), ts, ab, ex)
        c = np.bincount(g[tk & y], minlength=len(T))
        n = np.bincount(g[tk], minlength=len(T))
        out[(ts, ab)] = W.macro(c, n, T)[0]
    return out


def fit(tag, countries):
    from sklearn.linear_model import LogisticRegression
    ko, po, yo, so = val_pairs(ROOT, tag, "p2", countries)
    kt, pt, yt, st = val_pairs(TROOT, TTAG, "p", countries)
    u = np.union1d(ko, kt)
    io, ho = align(u, ko)
    it, ht = align(u, kt)
    P_o = np.where(ho, po[io], 0.0); P_t = np.where(ht, pt[it], 0.0)
    y = np.where(ho, yo[io], yt[it])
    s1 = np.where(ho, so[io], st[it])
    X = feats(P_o, ho.astype(float), P_t, ht.astype(float))
    lr = LogisticRegression(C=1.0, max_iter=500).fit(X, y)
    pb = lr.predict_proba(X)[:, 1]
    # every val entity of these countries, with its true match count
    gt = np.load(f"{ROOT}/work/cache/gt_train.npz")
    ids, ctry = s1_ids(ROOT, "train")
    fold = np.load(f"{ROOT}/work/models/val_scores_{tag}.npz")["fold"]
    ent = ((fold == W.VAL_FOLD) & np.isin(ctry, countries)).nonzero()[0]
    ent = ent[np.argsort(ids[ent])]
    T_row = np.zeros(len(ids), np.int32)
    np.add.at(T_row, gt["s1row"][gt["pq"]], 1)
    T = T_row[ent]
    g = np.searchsorted(ids[ent], s1)
    src = ((u >> 30) & 3).astype(np.int8); tid = (u & TIDMASK).astype(np.int64)
    print(f"[{'+'.join(countries)} / {tag}] union pairs {len(u):,} (ours {ho.sum():,}, "
          f"theirs {ht.sum():,}, both {(ho & ht).sum():,})  coef {np.round(lr.coef_[0], 3)}",
          flush=True)
    res = {}
    for name, p in (("ours", P_o), ("theirs", P_t), ("blend", pb)):
        res[name] = macro_grid(g, p, y, src, tid, T)
        best = max(res[name], key=res[name].get)
        print(f"   {name:<7} best {best} {res[name][best]:.5f}   "
              f"t1.3/a4 {res[name][(1.3, 4.0)]:.5f}   t1.1/a1 {res[name][(1.1, 1.0)]:.5f}",
              flush=True)
    return lr, res


def apply(lr, country, our_dump):
    ids_o, ctry_o = s1_ids(ROOT, "test")
    ids_t, _ = s1_ids(TROOT, "test")
    zo, zt = np.load(f"{DUMPD}/{our_dump}.npz"), np.load(f"{DUMPD}/{TDUMP}.npz")
    want = np.sort(ids_o[ctry_o == country])

    def keyed(z, ids, pre):
        k = pkey(ids[z[pre + "row"]], z[pre + "src"], z[pre + "tid"])
        return k[np.isin(k >> 32, want)], np.isin(ids[z[pre + "row"]], want)
    ko, mo = keyed(zo, ids_o, ""); po = zo["p"][mo].astype(np.float64)
    kt, mt = keyed(zt, ids_t, ""); pt = zt["p"][mt].astype(np.float64)
    oo, ot = np.argsort(ko), np.argsort(kt)
    ko, po, kt, pt = ko[oo], po[oo], kt[ot], pt[ot]
    co = np.unique(keyed(zo, ids_o, "c_")[0])
    ct = np.unique(keyed(zt, ids_t, "c_")[0])
    cand = np.union1d(co, ct)
    u = np.union1d(ko, kt)                        # pairs with p >= 0.02 in either model
    io, ho = align(u, ko)
    it, ht = align(u, kt)
    P_o = np.where(ho, po[io], 0.0); P_t = np.where(ht, pt[it], 0.0)
    pb = lr.predict_proba(feats(P_o, np.isin(u, co).astype(float),
                                P_t, np.isin(u, ct).astype(float)))[:, 1]
    o_ids = np.argsort(ids_o)                     # back to ber2 test row space
    to_row = lambda s1id: o_ids[np.searchsorted(ids_o[o_ids], s1id)]  # noqa: E731
    keep = pb >= FLOOR
    np.savez(f"{DUMPD}/blend_{country}.npz",
             c_row=to_row(cand >> 32).astype(np.int32),
             c_tid=(cand & TIDMASK).astype(np.int32),
             c_src=((cand >> 30) & 3).astype(np.int8),
             row=to_row(u[keep] >> 32).astype(np.int32),
             tid=(u[keep] & TIDMASK).astype(np.int32),
             src=((u[keep] >> 30) & 3).astype(np.int8), p=pb[keep].astype(np.float32))
    print(f"   test {country}: {len(cand):,} union candidates, {keep.sum():,} pairs p>=0.02 "
          f"(ours-only {(ho & ~ht).sum():,}, theirs-only {(ht & ~ho).sum():,})", flush=True)


def main():
    for country, tag, dump in OURS:
        lr, _ = fit(tag, ["US", "India"] if country == "France" else [country])
        apply(lr, country, dump)


if __name__ == "__main__":
    main()
