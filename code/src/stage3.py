"""
Stage 3: a LightGBM decision model over the cross-encoder scores, with the
context an independent per-pair blend cannot see -- where a pair ranks inside
its S1 entity, the best / runner-up scores there, how many strong candidates
the entity has, and how hard other S1 entities compete for the same target.

It works on the band the CE already covers (p2 >= 0.02, BAND_TAG=_all), so the
candidate set is unchanged.  Validation is honest: 5-fold cross-fitting over
the val-fold entities (no entity is scored by a model that saw it).  For test,
one model is fitted on all val pairs and applied to the test band.

Env: VAL_TAG, DUMPS, CE_TAGS (",2,3,4"), EVAL_CN, OUT_TAG (suffix of outputs).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import waterfall as W  # noqa: E402
import xenc as X  # noqa: E402

VAL_TAG = os.environ.get("VAL_TAG", "s2xUS_P2")
DUMPS = [d for d in os.environ.get("DUMPS", "us_p2").split(",") if d]
CE_TAGS = os.environ.get("CE_TAGS", ",2,3,4").split(",")
OT = os.environ.get("OUT_TAG", "")
BT = "_all"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, verbosity=-1, num_threads=20)
NROUND = int(os.environ.get("S3_NROUND", "400"))


def group_stats(g, s):
    """Per pair: rank of s inside its group g, group max, runner-up, margin to
    the best *other* member, group size, and members with s > 0."""
    o = np.lexsort((-s, g))
    gs, ss = g[o], s[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    rank = np.arange(len(gs)) - start
    _, inv, size = np.unique(gs, return_inverse=True, return_counts=True)
    gmax = ss[start]
    nxt = np.minimum(start + 1, len(ss) - 1)
    second = np.where(size[inv] > 1, ss[nxt], -20.0)
    margin = np.where(rank == 0, ss - second, ss - gmax)
    npos = np.bincount(inv, weights=(ss > 0))[inv]
    out = np.empty((len(g), 6), np.float32)
    out[o] = np.stack([rank, gmax, second, margin, size[inv], npos], 1)
    return out


USE_SRC = os.environ.get("S3_SRC", "0") == "1"        # source-balance features
USE_ANCHOR = os.environ.get("S3_ANCHOR", "0") == "1"  # transitive CE (anchor_export.py)


def source_balance(ent, src, cm):
    """Confident (CE > 0) matches of the entity in this pair's source and in the
    other one: an entity typically has matches in both S2 and S3."""
    u, inv = np.unique(ent, return_inverse=True)
    conf = np.zeros((len(u), 4)); np.add.at(conf, (inv, src.astype(int)), cm > 0)
    same = conf[inv, src.astype(int)]
    other = conf[inv, 5 - src.astype(int)]
    return np.stack([src, same, other, other - same], 1)


def anchor_feats(band):
    R = f"{X.ROOT}/work/runpod"
    j = np.load(f"{R}/anchor_map_{band}.npy")
    lg = np.load(f"{R}/logits_anchor.npy")
    return np.stack([np.where(j >= 0, lg[np.maximum(j, 0)], -20.0), (j >= 0)], 1)


def features(ent, tgt, lp2, ce, src=None, band=None):
    cm = ce.mean(1)
    cols = [lp2[:, None], ce, cm[:, None],
            group_stats(ent, cm), group_stats(ent, lp2)[:, [0, 3]],
            group_stats(tgt, cm)[:, [0, 1, 2, 3, 4]]]
    if USE_SRC:
        cols.append(source_balance(ent, src, cm))
    if USE_ANCHOR:
        cols.append(anchor_feats(band))
    return np.hstack(cols).astype(np.float32)


def ce_matrix(fmt):
    return np.stack([np.load(fmt.format(t=t)) for t in CE_TAGS], 1)


def main():
    import lightgbm as lgb
    v = np.load(f"{X.ROOT}/work/models/val_scores_{VAL_TAG}.npz")
    idx = np.load(f"{X.OUT}/val_band{BT}_{VAL_TAG}.npz")["idx"]
    ce = ce_matrix(f"{X.OUT}/ce{{t}}_val_band{BT}_{VAL_TAG}.npy")
    lp2 = X._logit(v["p2"][idx])
    ent = v["gt"][idx].astype(np.int64)
    tgt = (v["src"][idx].astype(np.int64) << 32) | v["tid"][idx].astype(np.int64)
    Xv = features(ent, tgt, lp2, ce, v["src"][idx], f"val_band{BT}_{VAL_TAG}")
    y = v["pos"][idx].astype(np.int8)
    fold = ((ent * 2654435761) >> 7) % 5
    oof = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        b = lgb.train(PARAMS, lgb.Dataset(Xv[tr], label=y[tr]), num_boost_round=NROUND)
        oof[te] = b.predict(Xv[te])
    p3 = v["p2"].astype(np.float64).copy()
    p3[idx] = oof
    tag3 = f"{VAL_TAG}s3{OT}"
    np.savez(f"{X.ROOT}/work/models/val_scores_{tag3}.npz",
             **{k: v[k] for k in v.files if k != "p2"}, p2=p3.astype(np.float32))
    for cn in os.environ.get("EVAL_CN", "US").split(","):
        os.environ["WF_COUNTRIES"] = cn
        V = W.load_val(tag3, "p2")
        g, p, yy, T = V["g"], V["p"], V["y"], V["T"]
        ex = W.exclusivity(p, V["src"], V["tid"])
        res = []
        for ts, ab in ((1.0, 1.0), (1.1, 1.0), (1.1, 2.0), (1.1, 4.0), (1.3, 1.0)):
            tk, _ = W.decide_pairs(g, p, len(T), ts, ab, ex)
            c = np.bincount(g[tk & yy], minlength=len(T))
            n = np.bincount(g[tk], minlength=len(T))
            res.append(f"t{ts}/a{ab} {W.macro(c, n, T)[0]:.5f}")
        print(f"  {tag3:<16} {cn:<6} " + "  ".join(res), flush=True)

    full = lgb.train(PARAMS, lgb.Dataset(Xv, label=y), num_boost_round=NROUND)
    for name in DUMPS:
        z = np.load(f"{X.DUMPD}/{name}.npz")
        tidx = np.load(f"{X.OUT}/test_band{BT}_{name}.npz")["idx"]
        cet = ce_matrix(f"{X.OUT}/ce{{t}}_test_band{BT}_{name}.npy")
        tent = z["row"][tidx].astype(np.int64)
        ttgt = (z["src"][tidx].astype(np.int64) << 32) | z["tid"][tidx].astype(np.int64)
        Xt = features(tent, ttgt, X._logit(z["p"][tidx]), cet, z["src"][tidx],
                      f"test_band{BT}_{name}")
        p = z["p"].astype(np.float64).copy()
        p[tidx] = full.predict(Xt)
        np.savez(f"{X.DUMPD}/{name}s3{OT}.npz", **{k: z[k] for k in z.files if k != "p"},
                 p=p.astype(np.float32))
        print(f"  wrote {name}s3{OT}.npz  mean p on band {z['p'][tidx].mean():.3f} -> "
              f"{p[tidx].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
