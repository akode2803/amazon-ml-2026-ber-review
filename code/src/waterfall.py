"""
Error waterfall on the validation fold: where does every true pair die, and
how much macro-F0.5 would fixing each stage buy?

Stages, in pipeline order, for every true (S1, target) pair of a val entity:

    lost in blocking        never retrieved as a candidate
    lost to min_p           retrieved, but p < MIN_P (dropped before decision)
    lost to exclusivity     another S1 entity claimed the target with higher p
    lost to abstention      the entity was predicted empty as a whole
    lost to prefix cut      entity predicted non-empty, pair not in the prefix
    TP                      predicted

Headroom is reported as the macro-F0.5 obtained when one stage is repaired by
an oracle while everything else stays as it is.  That is the number that says
where engineering time should go.

Usage:  python waterfall.py [ctag] [prob_key]
"""
from __future__ import annotations

import json
import os
import sys

import numba
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
MODELS = f"{ROOT}/work/models"
REPORTS = f"{ROOT}/work/reports"
VAL_FOLD = int(os.environ.get("VAL_FOLD", "0"))
MIN_P = 0.02


def load_val(ctag, pkey="p"):
    z = np.load(f"{MODELS}/val_scores_{ctag}.npz")
    gt, p, pos, src, tid = z["gt"], z[pkey], z["pos"], z["src"], z["tid"]
    T, fold, g2gt = z["T"], z["fold"], z["g2gt"]
    ctry = pl.read_parquet(f"{CACHE}/train/s1/meta.parquet",
                           columns=["country"])["country"].to_numpy()
    inv = np.empty(len(T), np.int64)
    inv[g2gt] = np.arange(len(g2gt))
    sel = fold == VAL_FOLD
    wfc = [c for c in os.environ.get("WF_COUNTRIES", "").split(",") if c]
    if wfc:                       # score a per-country expert on its own entities
        sel &= np.isin(ctry, wfc)
    val_gt = np.sort(g2gt[sel.nonzero()[0]])
    remap = np.full(len(T), -1, np.int64)
    remap[val_gt] = np.arange(len(val_gt))
    keep = remap[gt] >= 0
    return dict(g=remap[gt[keep]], p=p[keep].astype(np.float64),
                y=pos[keep].astype(bool), src=src[keep], tid=tid[keep],
                T=T[val_gt].astype(np.int32), ctry=ctry[inv[val_gt]])


def exclusivity(p, src, tid):
    """True where this pair is its target's most confident claimant."""
    key = src.astype(np.int64) * (int(tid.max()) + 2) + tid
    o = np.lexsort((-p, key))
    k2 = key[o]
    first = np.zeros(len(p), bool)
    first[o[np.concatenate([[True], k2[1:] != k2[:-1]])]] = True
    return first


def decide_pairs(g, p, n_ent, tscale, abstain, excl_mask):
    """Run the production decision; returns a per-pair 'taken' mask."""
    alive = excl_mask & (p >= MIN_P)
    idx = alive.nonzero()[0]
    o = idx[np.lexsort((-p[idx], g[idx]))]
    cnt = np.bincount(g[o], minlength=n_ent)
    ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cnt)])
    take = np.zeros(n_ent, np.int32)
    val = np.zeros(n_ent, np.float64)
    decide.decide_all(ptr, p[o], tscale, abstain, MIN_P, take, val)
    taken = np.zeros(len(p), bool)
    _mark(ptr, take, o, taken)
    return taken, take


@numba.njit(cache=True)
def _mark(ptr, take, o, taken):
    for gi in range(ptr.shape[0] - 1):
        for j in range(take[gi]):
            taken[o[ptr[gi] + j]] = True


def macro(c, n, T):
    per = np.where(n == 0, (T == 0).astype(np.float64),
                   np.where((T == 0) | (c == 0), 0.0,
                            1.25 * c / np.maximum(0.25 * T + n, 1e-9)))
    return per.mean(), per


def main():
    ctag = sys.argv[1] if len(sys.argv) > 1 else "c1"
    pkey = sys.argv[2] if len(sys.argv) > 2 else "p"
    V = load_val(ctag, pkey)
    g, p, y, T = V["g"], V["p"], V["y"], V["T"]
    n_ent = len(T)
    excl = exclusivity(p, V["src"], V["tid"])

    # pick the production operating point on this fold (small grid)
    best = None
    for ts in (1.0, 1.15, 1.3):
        for ab in (1.0, 2.0, 4.0, 8.0):
            taken, _ = decide_pairs(g, p, n_ent, ts, ab, excl)
            c = np.bincount(g[taken & y], minlength=n_ent)
            n = np.bincount(g[taken], minlength=n_ent)
            f, _ = macro(c, n, T)
            if best is None or f > best[0]:
                best = (f, ts, ab, taken)
    f0, ts, ab, taken = best
    print(f"[{ctag}/{pkey}] val entities {n_ent:,}  pairs {len(g):,}  "
          f"decision t={ts} a={ab}  macro-F0.5 {f0:.5f}")

    c = np.bincount(g[taken & y], minlength=n_ent)
    n = np.bincount(g[taken], minlength=n_ent)
    ent_empty = n == 0

    # ---------------------------------------------------- pair waterfall
    tot_true = int(T.sum())
    found = np.bincount(g[y], minlength=n_ent)
    lost_block = tot_true - int(found.sum())
    yp = y.copy()
    lost_minp = int((yp & (p < MIN_P)).sum())
    yp &= p >= MIN_P
    lost_excl = int((yp & ~excl).sum())
    yp &= excl
    lost_abst = int((yp & ent_empty[g]).sum())
    yp &= ~ent_empty[g]
    lost_cut = int((yp & ~taken).sum())
    tp = int((y & taken).sum())
    assert tp == int((yp & taken).sum())
    fp = int((taken & ~y).sum())
    rows = [("all true pairs", tot_true), ("lost in blocking", lost_block),
            ("lost to min_p (p<0.02)", lost_minp),
            ("lost to exclusivity", lost_excl),
            ("lost to entity abstention", lost_abst),
            ("lost to prefix cut / threshold", lost_cut),
            ("TRUE POSITIVES", tp), ("false positives", fp)]
    print("\n  pair waterfall")
    for k, v in rows:
        print(f"    {k:<34} {v:>10,}  {v / tot_true * 100:6.2f}%")
    prec = tp / max(tp + fp, 1)
    print(f"    micro precision {prec:.5f}  micro recall {tp / tot_true:.5f}")

    # --------------------------------------------- oracle-repair headroom
    def fix(add_c=None, drop_fp=False):
        cc = c.copy(); nn = n.copy()
        if drop_fp:
            nn = cc.copy()
        if add_c is not None:
            cc = cc + add_c; nn = nn + add_c
        return macro(cc, nn, T)[0]

    miss_block = T - found
    m_minp = np.bincount(g[y & (p < MIN_P)], minlength=n_ent)
    m_excl = np.bincount(g[y & (p >= MIN_P) & ~excl], minlength=n_ent)
    m_dec = np.bincount(g[y & (p >= MIN_P) & excl & ~taken], minlength=n_ent)
    fp_ent = n - c
    print("\n  oracle headroom (repair ONE stage, keep the rest):")
    heads = {
        "baseline": f0,
        "no false positives": fix(drop_fp=True),
        "perfect blocking (+found & taken)": fix(add_c=miss_block),
        "no min_p loss": fix(add_c=m_minp),
        "no exclusivity loss": fix(add_c=m_excl),
        "no decision loss (abstain+cut)": fix(add_c=m_dec),
        "perfect post-blocking (oracle ceiling)": macro(found, found, T)[0],
    }
    for k, v in heads.items():
        print(f"    {k:<40} {v:.5f}  (+{v - f0:.5f})")

    # FP families at the entity level
    sing = T == 0
    print("\n  where the F0.5 is lost (sum of 1-F per entity / n_ent):")
    _, per = macro(c, n, T)
    loss = 1 - per
    buckets = {
        "singleton false merge": sing & (n > 0),
        "T>0, predicted empty": (~sing) & (n == 0),
        "T>0, non-empty, 0 correct": (~sing) & (n > 0) & (c == 0),
        "T>0, some FP (c>0)": (~sing) & (c > 0) & (fp_ent > 0),
        "T>0, no FP, partial recall": (~sing) & (c > 0) & (fp_ent == 0) & (c < T),
    }
    for k, m in buckets.items():
        print(f"    {k:<32} n={m.sum():>8,}  loss {loss[m].sum() / n_ent:.5f}")
    print(f"    {'TOTAL':<32} {'':>10}  loss {loss.sum() / n_ent:.5f}")

    print("\n  by country:")
    for cn in sorted(set(V["ctry"].tolist())):
        m = V["ctry"] == cn
        print(f"    {cn:<8} n={m.sum():>8,}  F0.5 {per[m].mean():.5f}")
    print("\n  by T:")
    for t in range(0, 9):
        m = T == t
        if m.any():
            print(f"    T={t}  n={m.sum():>8,}  F0.5 {per[m].mean():.5f}  "
                  f"loss share {loss[m].sum() / loss.sum() * 100:5.1f}%")

    os.makedirs(REPORTS, exist_ok=True)
    with open(f"{REPORTS}/waterfall_{ctag}_{pkey}.json", "w") as fh:
        json.dump({"f05": f0, "tscale": ts, "abstain": ab,
                   "waterfall": dict(rows), "headroom": heads}, fh, indent=2)


if __name__ == "__main__":
    main()
