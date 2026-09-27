"""
End-to-end train + validate on the training split.

Entity-level 5-fold split (fold 0 = validation).  Everything downstream of the
split -- candidate generation, idf tables, normalisation -- is computed exactly
as it will be at test time, from the split's own records only, so the
validation number is an honest estimate of leaderboard behaviour.

Outputs: a trained model, calibrated probabilities, validation macro-F0.5 under
several decision rules (for the ablation), and per-slice diagnostics.
"""
from __future__ import annotations

import gc
import json
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
OUT = f"{ROOT}/work/models"
os.makedirs(OUT, exist_ok=True)

SPLIT = os.environ.get("SPLIT", "train")
CTAG = os.environ.get("CTAG", "c1")
MTAG = os.environ.get("MTAG", CTAG)   # output name; lets LOCO runs coexist
VAL_FOLD = int(os.environ.get("VAL_FOLD", "0"))
MAX_TRAIN_PER_PART = int(os.environ.get("MAX_TRAIN_PER_PART", "1200000"))
NROUND = int(os.environ.get("NROUND", "500"))
TRAIN_COUNTRIES = os.environ.get("TRAIN_COUNTRIES", "").split(",") if \
    os.environ.get("TRAIN_COUNTRIES") else None
PSEUDO = os.environ.get("PSEUDO", "")  # "India:locoUS:0.97:0.02" -> self-training proxy (true labels only audited)


def main():
    t00 = time.time()
    s1row, pq, ps, tgt_row = M.gt_arrays(SPLIT)
    n_gt = len(s1row)
    meta1 = pl.read_parquet(f"{CACHE}/{SPLIT}/s1/meta.parquet",
                            columns=["id", "country"])
    ids1 = meta1["id"].to_numpy()
    ctry1 = meta1["country"].to_numpy()
    fold = M.fold_of(ids1)
    g2gt = np.full(len(ids1), -1, np.int64); g2gt[s1row] = np.arange(n_gt)
    T = np.zeros(n_gt, np.int32); np.add.at(T, pq, 1)
    countries = M.countries_of(SPLIT)
    print(f"countries: {countries}   val fold {VAL_FOLD} = "
          f"{(fold==VAL_FOLD).mean()*100:.1f}% of entities", flush=True)

    # ---------------------------------------------------------------- idf
    idfs = {}
    for c in countries:
        t0 = time.time()
        idfs[c] = M.idf_tables(SPLIT, c)
        print(f"  idf[{c}] ready in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------- training
    train_folds = [f for f in range(5) if f != VAL_FOLD]
    tc = TRAIN_COUNTRIES or countries
    print(f"\nbuilding training set from folds {train_folds} "
          f"countries {tc}", flush=True)
    nf = len(F.feature_names())
    ps_c = PSEUDO.split(":")[0].split(",") if PSEUDO else []
    nparts = (len(tc) + len(ps_c)) * 2
    # preallocate: concatenating per-partition blocks would transiently double
    # a multi-GB matrix, which this box (16 GB) cannot absorb
    X = np.empty((nparts * MAX_TRAIN_PER_PART, nf), np.float32)
    y = np.empty(nparts * MAX_TRAIN_PER_PART, np.int8)
    filled = 0
    for country in tc:
        for source in (2, 3):
            t0 = time.time()
            part = M.Partition(SPLIT, country, source, CTAG, idfs[country])
            lab, _ = part.labels(s1row, pq, ps, tgt_row)
            qfold = fold[part.rows1][part.q]
            sel = np.isin(qfold, train_folds).nonzero()[0]
            rng = np.random.default_rng(17)
            if len(sel) > MAX_TRAIN_PER_PART:
                sel = np.sort(rng.choice(sel, MAX_TRAIN_PER_PART, replace=False))
            for a in range(0, len(sel), 1_000_000):
                b = min(a + 1_000_000, len(sel))
                sub = sel[a:b]
                X[filled + a:filled + b] = F.compute(
                    part.qs, part.cs, part.q[sub], part.c[sub], part.ctx[sub])
            y[filled:filled + len(sel)] = lab[sel]
            filled += len(sel)
            print(f"  train {country:7s} S{source}: {len(sel):>9,} pairs "
                  f"({lab[sel].mean()*100:5.2f}% pos)  {time.time()-t0:.0f}s",
                  flush=True)
            part.free(); del part, lab
            F._HMEMO.clear(); gc.collect()
    if PSEUDO:
        import lightgbm as lgb
        _, ptag, hi, lo = PSEUDO.split(":")
        hi, lo = float(hi), float(lo)
        pb = lgb.Booster(model_file=f"{OUT}/lgbm_{ptag}.txt")
        for country in ps_c:
            for source in (2, 3):
                t0 = time.time()
                part = M.Partition(SPLIT, country, source, CTAG, idfs[country])
                lab, _ = part.labels(s1row, pq, ps, tgt_row)   # audit only
                rng = np.random.default_rng(29)
                sel = np.arange(len(part.q))
                if len(sel) > 2 * MAX_TRAIN_PER_PART:
                    sel = np.sort(rng.choice(sel, 2 * MAX_TRAIN_PER_PART,
                                             replace=False))
                Xp = np.empty((len(sel), nf), np.float32)
                for a in range(0, len(sel), 1_000_000):
                    sub = sel[a:a + 1_000_000]
                    Xp[a:a + len(sub)] = F.compute(part.qs, part.cs, part.q[sub],
                                                   part.c[sub], part.ctx[sub])
                pp = pb.predict(Xp)
                ok = (pp >= hi) | (pp <= lo)
                idx = ok.nonzero()[0]
                if len(idx) > MAX_TRAIN_PER_PART:
                    idx = np.sort(rng.choice(idx, MAX_TRAIN_PER_PART, replace=False))
                X[filled:filled + len(idx)] = Xp[idx]
                yl = (pp[idx] >= hi).astype(np.int8)
                y[filled:filled + len(idx)] = yl
                filled += len(idx)
                tl = lab[sel][idx]
                print(f"  pseudo {country:7s} S{source}: {len(idx):>9,} pairs "
                      f"({yl.mean()*100:5.2f}% pos)  pseudo-pos precision "
                      f"{tl[yl == 1].mean():.4f}  pseudo-neg purity "
                      f"{1 - tl[yl == 0].mean():.5f}  {time.time()-t0:.0f}s",
                      flush=True)
                part.free(); del part, lab, Xp
                F._HMEMO.clear(); gc.collect()
    X = X[:filled]; y = y[:filled]
    gc.collect()
    print(f"training matrix {X.shape}  {X.nbytes/1e9:.2f} GB  "
          f"pos rate {y.mean()*100:.2f}%", flush=True)

    # hold out a slice of the TRAINING folds for early stopping / calibration
    rng = np.random.default_rng(3)
    hold = rng.random(len(y)) < 0.10
    t0 = time.time()
    booster = M.train_lgbm(X[~hold], y[~hold], num_round=NROUND,
                           valid=(X[hold], y[hold]))
    print(f"lgbm trained in {time.time()-t0:.0f}s, "
          f"{booster.num_trees()} trees", flush=True)

    # ---- isotonic calibration on the held-out slice
    from sklearn.isotonic import IsotonicRegression
    raw = booster.predict(X[hold], num_iteration=booster.best_iteration or NROUND)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y[hold])
    booster.save_model(f"{OUT}/lgbm_{MTAG}.txt")
    with open(f"{OUT}/iso_{MTAG}.pkl", "wb") as fh:
        pickle.dump(iso, fh)
    # reliability of raw vs calibrated
    print("\ncalibration (held-out slice of training folds):")
    print(f"  {'bin':>12} {'n':>9} {'mean raw':>9} {'mean cal':>9} {'actual':>8}")
    cal = iso.predict(raw)
    for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
        m = (raw >= lo) & (raw < hi)
        if m.sum():
            print(f"  [{lo:.1f},{hi:.1f}) {m.sum():>9,} {raw[m].mean():>9.4f} "
                  f"{cal[m].mean():>9.4f} {y[hold][m].mean():>8.4f}")
    imp = sorted(zip(F.feature_names(), booster.feature_importance("gain")),
                 key=lambda t: -t[1])
    print("\ntop-25 features by gain:")
    for n, v in imp[:25]:
        print(f"  {n:<14} {v:,.0f}")
    del X, y, raw, cal; gc.collect()

    # ------------------------------------------------------------ validation
    print("\nscoring validation fold ...", flush=True)
    V = {"gt": [], "p": [], "pos": [], "src": [], "tid": [],
         "fr": [], "rr": []}
    for country in countries:
        for source in (2, 3):
            t0 = time.time()
            part = M.Partition(SPLIT, country, source, CTAG, idfs[country])
            lab, _ = part.labels(s1row, pq, ps, tgt_row)
            qfold = fold[part.rows1][part.q]
            sel = (qfold == VAL_FOLD).nonzero()[0]
            probs = np.empty(len(sel), np.float32)
            for a in range(0, len(sel), 2_000_000):
                b = min(a + 2_000_000, len(sel))
                sub = sel[a:b]
                Xc = F.compute(part.qs, part.cs, part.q[sub], part.c[sub],
                               part.ctx[sub])
                probs[a:b] = iso.predict(booster.predict(
                    Xc, num_iteration=booster.best_iteration or NROUND))
                del Xc
            tgt_ids = pl.read_parquet(f"{CACHE}/{SPLIT}/s{source}/meta.parquet",
                                      columns=["id"])["id"].to_numpy()
            V["gt"].append(g2gt[part.rows1][part.q[sel]].astype(np.int64))
            V["p"].append(probs)
            V["pos"].append(lab[sel])
            V["src"].append(np.full(len(sel), source, np.int8))
            V["tid"].append(tgt_ids[part.rowsT][part.c[sel]].astype(np.int32))
            V["fr"].append(part.frank[sel]); V["rr"].append(part.rrank[sel])
            print(f"  val {country:7s} S{source}: {len(sel):>9,} pairs  "
                  f"{time.time()-t0:.0f}s", flush=True)
            part.free(); del part, lab, probs
            F._HMEMO.clear(); gc.collect()

    gtv = np.concatenate(V["gt"]); pv = np.concatenate(V["p"])
    posv = np.concatenate(V["pos"]); srcv = np.concatenate(V["src"])
    tidv = np.concatenate(V["tid"])
    frv = np.concatenate(V["fr"]); rrv = np.concatenate(V["rr"])
    del V; gc.collect()
    np.savez(f"{OUT}/val_scores_{MTAG}.npz", gt=gtv, p=pv, pos=posv,
             src=srcv, tid=tidv, fr=frv, rr=rrv, T=T, fold=fold, g2gt=g2gt)

    evaluate(gtv, pv, posv, T, fold, g2gt, ctry1, s1row, n_gt)
    print(f"\ntotal {time.time()-t00:.0f}s")


def evaluate(gtv, pv, posv, T, fold, g2gt, ctry1, s1row, n_gt):
    """Group validation pairs per entity and compare decision rules."""
    val_ent = (fold[s1row] == VAL_FOLD).nonzero()[0]     # GT rows in val
    val_ent = np.sort(g2gt[s1row[val_ent]])
    remap = np.full(n_gt, -1, np.int64)
    remap[val_ent] = np.arange(len(val_ent))
    keep = remap[gtv] >= 0
    gg = remap[gtv[keep]]; pp = pv[keep]; yy = posv[keep]
    order = np.lexsort((-pp, gg))
    gg, pp, yy = gg[order], pp[order], yy[order]
    pp64 = pp.astype(np.float64)          # convert once, not per grid point
    cnt = np.bincount(gg, minlength=len(val_ent))
    ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt)])
    Tv = T[val_ent].astype(np.int32)
    print(f"\nvalidation: {len(val_ent):,} entities, {len(gg):,} candidate pairs, "
          f"{Tv.sum():,} true matches, {(Tv==0).mean()*100:.2f}% singletons")
    print(f"  candidate recall on val: {yy.sum()/Tv.sum()*100:.3f}%")

    take = np.zeros(len(val_ent), np.int32); val = np.zeros(len(val_ent), np.float64)
    best = (-1, None)
    print(f"\n  {'rule':<34} {'macro F0.5':>11} {'prec':>8} {'rec':>8} "
          f"{'F1':>8} {'abstain':>8}")
    for thr in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        s = decide.f05_threshold(ptr, pp, yy, Tv, thr)
        print(f"  {'threshold p>=' + str(thr):<34} {s:>11.5f}")
        if s > best[0]:
            best = (s, f"threshold {thr}")
    for thr in (0.3, 0.5, 0.7):
        s = decide.f05_top1(ptr, pp, yy, Tv, thr)
        print(f"  {'top-1 only, p>=' + str(thr):<34} {s:>11.5f}")
    results = {}
    for tscale in (1.0, 1.05, 1.1):
        for ab in (0.5, 0.75, 1.0, 1.5):
            decide.decide_all(ptr, pp64, tscale, ab, 0.02, take, val)
            s = decide.f05_from_take(ptr, yy, take, Tv)
            results[(tscale, ab)] = s
            tp = sum_correct(ptr, yy, take)
            npred = take.sum()
            prec = tp / max(npred, 1)
            rec = tp / max(Tv.sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"  {'expected-F0.5 t=' + str(tscale) + ' a=' + str(ab):<34} "
                  f"{s:>11.5f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} "
                  f"{(take==0).mean()*100:>7.2f}%")
            if s > best[0]:
                best = (s, f"expected-F0.5 t={tscale} a={ab}")
    print(f"\n  BEST: {best[1]}  macro-F0.5 = {best[0]:.5f}")
    with open(f"{OUT}/val_summary_{MTAG}.json", "w") as fh:
        json.dump({"best_rule": best[1], "best_f05": best[0],
                   "grid": {f"{k}": v for k, v in results.items()}}, fh, indent=2)


def sum_correct(ptr, yy, take):
    tot = 0
    for g in range(len(take)):
        lo = ptr[g]
        tot += int(yy[lo:lo + take[g]].sum())
    return tot


if __name__ == "__main__":
    main()
