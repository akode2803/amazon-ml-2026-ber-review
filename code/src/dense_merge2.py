"""
Dense-retrieval rescue for any country (generalises dense_merge.py).

The new-pair model is fitted on the val new pairs of one or more countries
(FIT="suffix:val_tag,..."), each evaluated honestly with 5-fold cross-fitting
over its entities, then refitted on all of them and applied to one test dump.
France has no labels, so its model is fitted on US + India val pairs -- the
features (large-CE logit, retriever similarity/rank, entity context) are the
same everywhere.

    FIT=_US:s2xUS_P2s3_c5 APPLY_SUF=_US DUMP=us_p2s3_c5 DK=3 python dense_merge2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_merge as DM  # noqa: E402
import model as M  # noqa: E402
import waterfall as W  # noqa: E402

R = DM.R
DK = int(os.environ.get("DK", "3"))
FIT = [x.split(":") for x in os.environ.get("FIT", "_US:s2xUS_P2s3_c5").split(",")]
APPLY_SUF = os.environ.get("APPLY_SUF", "_US")
DUMP = os.environ.get("DUMP", "us_p2s3_c5")
OUT = os.environ.get("OUT", "")


def load(suf):
    k = np.load(f"{R}/dense_new_keys{suf}.npz")
    lg = np.load(f"{R}/logits_dense_new{suf}.npy")
    m = k["rank"] < DK
    return {n: k[n][m] for n in k.files} | {"ce": lg[m]}


def val_block(suf, tag):
    d = load(suf)
    v = np.load(f"{M.ROOT}/work/models/val_scores_{tag}.npz")
    mv = d["split"] == 0
    ent = v["g2gt"][d["r1"][mv]].astype(np.int64)
    ok = ent >= 0
    vi = np.flatnonzero(mv)[ok]; ent = ent[ok]
    ids_t = {s: DM.ids_of("train", s) for s in (2, 3)}
    src = d["src"][vi]
    tid = np.where(src == 2, ids_t[2][np.minimum(d["rT"][vi], len(ids_t[2]) - 1)],
                   ids_t[3][np.minimum(d["rT"][vi], len(ids_t[3]) - 1)])
    ctx = DM.context(ent, src, v["gt"].astype(np.int64), v["src"], v["p2"].astype(np.float64))
    X = np.column_stack([d["ce"][vi], d["sim"][vi], d["rank"][vi], src, ctx]).astype(np.float32)
    return dict(X=X, y=d["y"][vi].astype(np.int8), ent=ent, src=src, tid=tid, v=v, tag=tag)


def main():
    import lightgbm as lgb
    blocks = [val_block(s, t) for s, t in FIT]
    for b in blocks:                                   # honest per-country evaluation
        fold = ((b["ent"] * 2654435761) >> 7) % 5
        oof = np.zeros(len(b["y"]))
        others = [o for o in blocks if o is not b]
        for f in range(5):
            tr = fold != f
            Xtr = np.vstack([b["X"][tr]] + [o["X"] for o in others])
            ytr = np.concatenate([b["y"][tr]] + [o["y"] for o in others])
            m = lgb.train(DM.PARAMS, lgb.Dataset(Xtr, label=ytr), num_boost_round=300)
            oof[fold == f] = m.predict(b["X"][fold == f])
        v = b["v"]
        out = {kk: v[kk] for kk in v.files}
        for key, add in (("gt", b["ent"]), ("p2", oof.astype(np.float32)),
                         ("p", np.zeros(len(oof))), ("pos", b["y"]), ("src", b["src"]),
                         ("tid", b["tid"])):
            out[key] = np.concatenate([v[key], add.astype(v[key].dtype)])
        tagd = f"{b['tag']}d{DK}{OUT}"
        np.savez(f"{M.ROOT}/work/models/val_scores_{tagd}.npz", **out)
        cn = "India" if "IN" in b["tag"] else "US"
        for tag in (b["tag"], tagd):
            os.environ["WF_COUNTRIES"] = cn
            V = W.load_val(tag, "p2")
            g, p, yy, T = V["g"], V["p"], V["y"], V["T"]
            ex = W.exclusivity(p, V["src"], V["tid"])
            tk, _ = W.decide_pairs(g, p, len(T), 1.1, 1.0, ex)
            c = np.bincount(g[tk & yy], minlength=len(T)); n = np.bincount(g[tk], minlength=len(T))
            print(f"  {tag:<26} {cn:<6} t1.1/a1.0 {W.macro(c, n, T)[0]:.5f}", flush=True)

    full = lgb.train(DM.PARAMS, lgb.Dataset(np.vstack([b["X"] for b in blocks]),
                                            label=np.concatenate([b["y"] for b in blocks])),
                     num_boost_round=300)
    d = load(APPLY_SUF)
    mt = np.flatnonzero(d["split"] == 1)
    z = np.load(f"{M.ROOT}/work/dumps/{DUMP}.npz")
    ids_tt = {s: DM.ids_of("test", s) for s in (2, 3)}
    src = d["src"][mt]
    ttid = np.where(src == 2, ids_tt[2][np.minimum(d["rT"][mt], len(ids_tt[2]) - 1)],
                    ids_tt[3][np.minimum(d["rT"][mt], len(ids_tt[3]) - 1)]).astype(np.int32)
    ctx = DM.context(d["r1"][mt], src, z["row"].astype(np.int64), z["src"], z["p"].astype(np.float64))
    pt = full.predict(np.column_stack([d["ce"][mt], d["sim"][mt], d["rank"][mt], src, ctx]).astype(np.float32))
    kp = pt >= 0.02
    r1 = d["r1"][mt]
    np.savez(f"{M.ROOT}/work/dumps/{DUMP}d{DK}{OUT}.npz",
             c_row=np.concatenate([z["c_row"], r1.astype(z["c_row"].dtype)]),
             c_tid=np.concatenate([z["c_tid"], ttid.astype(z["c_tid"].dtype)]),
             c_src=np.concatenate([z["c_src"], src.astype(z["c_src"].dtype)]),
             row=np.concatenate([z["row"], r1[kp].astype(z["row"].dtype)]),
             tid=np.concatenate([z["tid"], ttid[kp].astype(z["tid"].dtype)]),
             src=np.concatenate([z["src"], src[kp].astype(z["src"].dtype)]),
             p=np.concatenate([z["p"], pt[kp].astype(z["p"].dtype)]))
    print(f"test {APPLY_SUF}: {len(mt):,} new candidates, {kp.sum():,} with p>=0.02, "
          f"{(pt > 0.5).sum():,} with p>0.5 -> {DUMP}d{DK}{OUT}.npz", flush=True)


if __name__ == "__main__":
    main()
