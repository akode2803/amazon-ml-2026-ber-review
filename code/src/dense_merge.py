"""
India dense-retrieval rescue, step 4 (local): turn the new pairs (tuned
retriever top-DK, not in c2, scored by the large CE on RunPod) into
probabilities and merge them into India's val / test predictions.

p for a new pair comes from a small LightGBM over the large-CE logit, the
retriever similarity and rank, the source, and the entity's existing matches
(best current p, confident matches in each source) -- cross-fitted over val
entities for an honest val number, then refitted on all val pairs for test.

    DK=3 python dense_merge.py
Writes val_scores_{VAL_TAG}d{DK}.npz (for waterfall) and {DUMP}d{DK}.npz (for
assemble.py); new pairs join c_* (they are scored by the matching model).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
import waterfall as W  # noqa: E402

R = f"{M.ROOT}/work/runpod"
DK = int(os.environ.get("DK", "3"))
VAL_TAG = os.environ.get("VAL_TAG", "s2xIN_P2s3_c5")
DUMP = os.environ.get("DUMP", "in_p2s3_c5")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
              num_threads=20)


def ids_of(split, s):
    return pl.read_parquet(f"{M.CACHE}/{split}/s{s}/meta.parquet",
                           columns=["id"])["id"].to_numpy().astype(np.int64)


def context(ent_new, src_new, ent_old, src_old, p_old):
    """Entity context for each new pair: best existing p, and confident (p>0.5)
    existing matches in the new pair's source and in the other source."""
    u, inv = np.unique(np.concatenate([ent_new, ent_old]), return_inverse=True)
    inv_new, inv_old = inv[:len(ent_new)], inv[len(ent_new):]
    best = np.zeros(len(u)); np.maximum.at(best, inv_old, p_old)
    conf = np.zeros((len(u), 4)); np.add.at(conf, (inv_old, src_old.astype(int)), p_old > 0.5)
    same = conf[inv_new, src_new.astype(int)]
    other = conf[inv_new, 5 - src_new.astype(int)]
    return np.stack([best[inv_new], same, other], 1)


def main():
    import lightgbm as lgb
    k = np.load(f"{R}/dense_new_keys.npz")
    lg = np.load(f"{R}/logits_dense_new.npy")
    keep = k["rank"] < DK
    sp, r1, rT, src = k["split"][keep], k["r1"][keep], k["rT"][keep], k["src"][keep]
    sim, rank, y, ce = k["sim"][keep], k["rank"][keep], k["y"][keep], lg[keep]

    # ---------------------------------------------------------------- val
    v = np.load(f"{M.ROOT}/work/models/val_scores_{VAL_TAG}.npz")
    mv = sp == 0
    g2gt = v["g2gt"]
    ent_new = g2gt[r1[mv]].astype(np.int64)
    ok = ent_new >= 0
    ent_new = ent_new[ok]
    vi = np.flatnonzero(mv)[ok]
    ids_t = {s: ids_of("train", s) for s in (2, 3)}
    tid_new = np.where(src[vi] == 2, ids_t[2][np.minimum(rT[vi], len(ids_t[2]) - 1)],
                       ids_t[3][np.minimum(rT[vi], len(ids_t[3]) - 1)])
    ctx = context(ent_new, src[vi], v["gt"].astype(np.int64), v["src"], v["p2"].astype(np.float64))
    Xv = np.column_stack([ce[vi], sim[vi], rank[vi], src[vi], ctx]).astype(np.float32)
    yv = y[vi].astype(np.int8)
    fold = ((ent_new * 2654435761) >> 7) % 5
    oof = np.zeros(len(yv))
    for f in range(5):
        tr, te = fold != f, fold == f
        b = lgb.train(PARAMS, lgb.Dataset(Xv[tr], label=yv[tr]), num_boost_round=300)
        oof[te] = b.predict(Xv[te])
    out = {kk: v[kk] for kk in v.files}
    out["gt"] = np.concatenate([v["gt"], ent_new.astype(v["gt"].dtype)])
    out["p2"] = np.concatenate([v["p2"], oof.astype(np.float32)])
    out["p"] = np.concatenate([v["p"], np.zeros(len(vi), v["p"].dtype)])
    out["pos"] = np.concatenate([v["pos"], yv.astype(v["pos"].dtype)])
    out["src"] = np.concatenate([v["src"], src[vi].astype(v["src"].dtype)])
    out["tid"] = np.concatenate([v["tid"], tid_new.astype(v["tid"].dtype)])
    np.savez(f"{M.ROOT}/work/models/val_scores_{VAL_TAG}d{DK}.npz", **out)
    print(f"val new pairs {len(vi):,} (true {yv.sum():,}); OOF p>0.5: {(oof > 0.5).sum():,} "
          f"of which true {yv[oof > 0.5].sum():,}", flush=True)
    for tag in (VAL_TAG, f"{VAL_TAG}d{DK}"):
        os.environ["WF_COUNTRIES"] = "India"
        V = W.load_val(tag, "p2")
        g, p, yy, T = V["g"], V["p"], V["y"], V["T"]
        ex = W.exclusivity(p, V["src"], V["tid"])
        res = []
        for ts, ab in ((1.0, 1.0), (1.1, 1.0), (1.1, 2.0), (1.1, 4.0)):
            tk, _ = W.decide_pairs(g, p, len(T), ts, ab, ex)
            c = np.bincount(g[tk & yy], minlength=len(T))
            n = np.bincount(g[tk], minlength=len(T))
            res.append(f"t{ts}/a{ab} {W.macro(c, n, T)[0]:.5f}")
        print(f"  {tag:<22} India " + "  ".join(res), flush=True)

    # --------------------------------------------------------------- test
    full = lgb.train(PARAMS, lgb.Dataset(Xv, label=yv), num_boost_round=300)
    z = np.load(f"{M.ROOT}/work/dumps/{DUMP}.npz")
    mt = np.flatnonzero(sp == 1)
    ids_tt = {s: ids_of("test", s) for s in (2, 3)}
    ttid = np.where(src[mt] == 2, ids_tt[2][np.minimum(rT[mt], len(ids_tt[2]) - 1)],
                    ids_tt[3][np.minimum(rT[mt], len(ids_tt[3]) - 1)]).astype(np.int32)
    tctx = context(r1[mt], src[mt], z["row"].astype(np.int64), z["src"], z["p"].astype(np.float64))
    pt = full.predict(np.column_stack([ce[mt], sim[mt], rank[mt], src[mt], tctx]).astype(np.float32))
    kp = pt >= 0.02
    np.savez(f"{M.ROOT}/work/dumps/{DUMP}d{DK}.npz",
             c_row=np.concatenate([z["c_row"], r1[mt].astype(z["c_row"].dtype)]),
             c_tid=np.concatenate([z["c_tid"], ttid.astype(z["c_tid"].dtype)]),
             c_src=np.concatenate([z["c_src"], src[mt].astype(z["c_src"].dtype)]),
             row=np.concatenate([z["row"], r1[mt][kp].astype(z["row"].dtype)]),
             tid=np.concatenate([z["tid"], ttid[kp].astype(z["tid"].dtype)]),
             src=np.concatenate([z["src"], src[mt][kp].astype(z["src"].dtype)]),
             p=np.concatenate([z["p"], pt[kp].astype(z["p"].dtype)]))
    print(f"test new pairs {len(mt):,} added as candidates; {kp.sum():,} with p>=0.02; "
          f"{(pt > 0.5).sum():,} with p>0.5  -> {DUMP}d{DK}.npz", flush=True)


if __name__ == "__main__":
    main()
