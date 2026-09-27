"""
Label-shift correction for the test set's extra look-alike decoys.

Finding (26 Sep, after the 0.959 leaderboard score): test contains ~3x more
"competing-cluster" candidates per S1 entity than train -- sibling businesses
with a near-identical name, on the same street, at a *different* building
number that several of their own records share ("Interstate Index Lakeside |
14 Waldingfield Rd" next to the true "Interstate Index | 5 Waldingfield Rd").
They have no S1 record, so every one we accept is a false positive.  The model
learned P(match | pattern) at train's decoy rate and over-accepts them.

Correction (Saerens, Latinne & Decaestecker 2002): inside each segment the
class-conditional feature distribution is assumed stable and only the prior
moves.  The test prior of a segment is re-estimated by EM on the model's own
calibrated posteriors (no labels), and every posterior is rescaled by the odds
ratio.  Segments (per candidate pair, from building-number relations):

    0 num_in_s1    a candidate number equals an S1 number
    1 competing    candidate numbers absent from S1, shared by >=1 other
                   candidate of the entity, while the entity also has a
                   candidate matching an S1 number (two rival clusters)
    2 lone_alt     as 1 but no candidate matches an S1 number
    3 alt_unsup    candidate numbers absent from S1, unsupported
    4 no_num       candidate has no building number

Usage:
    python shiftfix.py val     # prior table + injected-decoy simulation
    python shiftfix.py test    # corrected dumps -> work/dumps/*_sf.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import model as M  # noqa: E402
import waterfall as W  # noqa: E402

ROOT = M.ROOT
CACHE = f"{ROOT}/work/cache"
CTAG = os.environ.get("CTAG", "c2")
NSEG = 5
SEG_NAMES = ["num_in_s1", "competing", "lone_alt", "alt_unsup", "no_num"]
MIN_P = 0.02


def _explode(ptr, val, idx):
    lens = (ptr[idx + 1] - ptr[idx]).astype(np.int64)
    rep = np.repeat(np.arange(len(idx)), lens)
    start = np.repeat(ptr[idx], lens)
    off = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    return rep, val[start + off].astype(np.int64)


def segments(split, country):
    """Segment code for every candidate pair of one country.

    Returns (global S1 row, src, target id, seg) arrays."""
    idft = M.idf_tables(split, country)
    qs = F.FieldStore(split, country, 1, idft)
    Q, C, S, TID, stores = [], [], [], [], {}
    for s in (2, 3):
        z = np.load(f"{CACHE}/{split}/cand/{CTAG}_{country}_s{s}.npz")
        stores[s] = F.FieldStore(split, country, s, idft)
        ids = pl.read_parquet(f"{CACHE}/{split}/s{s}/meta.parquet",
                              columns=["id"])["id"].to_numpy()
        Q.append(z["q"]); C.append(z["c"])
        S.append(np.full(len(z["q"]), s, np.int8))
        TID.append(ids[z["rowsT"]][z["c"]].astype(np.int32))
        rows1 = z["rows1"]
    q = np.concatenate(Q).astype(np.int64); c = np.concatenate(C).astype(np.int64)
    src = np.concatenate(S); tid = np.concatenate(TID)
    n = len(q)
    reps, hs = [], []
    for s, st in stores.items():
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
    _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    sup = cnt[inv] - 1
    alt = np.zeros(n, np.int32)
    np.maximum.at(alt, pi[~inq], sup[~inq])
    hasnum = np.zeros(n, bool); hasnum[pi] = True
    anyin = np.zeros(n, bool); anyin[pi[inq]] = True
    ent_same = np.zeros(qs.n, bool); ent_same[q[anyin]] = True
    seg = np.full(n, 3, np.int8)
    seg[~hasnum] = 4
    seg[anyin] = 0
    rest = hasnum & ~anyin & (alt >= 1)
    seg[rest & ent_same[q]] = 1
    seg[rest & ~ent_same[q]] = 2
    return rows1[q].astype(np.int64), src, tid, seg


def lookup(keys_sorted, seg_sorted, row, src, tid):
    k = (row.astype(np.int64) * 4 + src) * (1 << 31) + tid
    j = np.clip(np.searchsorted(keys_sorted, k), 0, len(keys_sorted) - 1)
    ok = keys_sorted[j] == k
    return np.where(ok, seg_sorted[j], 4).astype(np.int8), ok


def seg_index(split, countries):
    R, S_, T_, G = [], [], [], []
    for c in countries:
        r, s, t, g = segments(split, c)
        R.append(r); S_.append(s); T_.append(t); G.append(g)
    r = np.concatenate(R); s = np.concatenate(S_); t = np.concatenate(T_)
    g = np.concatenate(G)
    k = (r * 4 + s) * (1 << 31) + t
    o = np.argsort(k)
    return k[o], g[o]


def em_prior(p, pi_tr, iters=200):
    """Saerens EM: test prior of a segment from calibrated posteriors."""
    pi = pi_tr
    for _ in range(iters):
        w1, w0 = pi / pi_tr, (1 - pi) / (1 - pi_tr)
        q = w1 * p / (w1 * p + w0 * (1 - p))
        new = q.mean()
        if abs(new - pi) < 1e-7:
            break
        pi = new
    return pi


def adjust(p, pi_tr, pi_te):
    w1, w0 = pi_te / pi_tr, (1 - pi_te) / (1 - pi_tr)
    return w1 * p / (w1 * p + w0 * (1 - p))


def val_priors(p, y, seg):
    """Train-side segment prior over the decision-relevant pairs (p>=MIN_P).
    Uses the mean calibrated p (what the model believes) as the reference prior,
    so that EM on unshifted data is a fixed point."""
    out = {}
    for g in range(NSEG):
        m = (seg == g) & (p >= MIN_P)
        out[g] = (float(p[m].mean()) if m.any() else 0.5, int(m.sum()),
                  float(y[m].mean()) if m.any() else 0.5)
    return out


def correct(p, seg, priors, label=""):
    q = p.copy()
    for g in range(NSEG):
        m = (seg == g) & (p >= MIN_P)
        if m.sum() < 100:
            continue
        pi_tr = priors[g][0]
        pi_te = em_prior(p[m], pi_tr)
        q[m] = adjust(p[m], pi_tr, pi_te)
        print(f"   {label}{SEG_NAMES[g]:<10} n={m.sum():>9,}  prior ref {pi_tr:.4f} "
              f"(true {priors[g][2]:.4f}) -> EM {pi_te:.4f}  "
              f"odds x{(pi_te / (1 - pi_te)) / (pi_tr / (1 - pi_tr)):.3f}")
    return q


def f05(V, p):
    g, y, T = V["g"], V["y"], V["T"]
    excl = W.exclusivity(p, V["src"], V["tid"])
    res = {}
    for ts, ab in ((1.1, 1.0), (1.1, 2.0), (1.1, 4.0)):
        taken, _ = W.decide_pairs(g, p, len(T), ts, ab, excl)
        c = np.bincount(g[taken & y], minlength=len(T))
        k = np.bincount(g[taken], minlength=len(T))
        res[f"a={ab}"] = round(W.macro(c, k, T)[0], 5)
    return res


def val_segments():
    V = W.load_val("moe", "p2")
    z = np.load(f"{ROOT}/work/models/val_scores_moe.npz")
    g2gt = z["g2gt"]; fold = z["fold"]
    inv = np.empty(len(g2gt), np.int64); inv[g2gt] = np.arange(len(g2gt))
    val_gt = np.sort(g2gt[(fold == 0).nonzero()[0]])
    rows = inv[val_gt][V["g"]]
    keys, segs = seg_index("train", ["India", "US"])
    seg, ok = lookup(keys, segs, rows, V["src"], V["tid"])
    pc = V["ctry"][V["g"]]
    pri = {cn: val_priors(V["p"][pc == cn], V["y"][pc == cn], seg[pc == cn])
           for cn in ("India", "US")}
    pri["France"] = val_priors(V["p"], V["y"], seg)          # joint model, pooled
    return V, seg, ok, pc, pri


def main_val():
    V, seg, ok, pc, pri = val_segments()
    print(f"val pairs matched to a segment: {ok.mean() * 100:.2f}%")
    for cn in ("India", "US"):
        print(f" {cn}: " + "  ".join(f"{SEG_NAMES[g]} meanp={pri[cn][g][0]:.3f} "
                                     f"true={pri[cn][g][2]:.3f} n={pri[cn][g][1]:,}"
                                     for g in range(NSEG)))
    p0 = V["p"].copy(); pcor = p0.copy()
    for cn in ("India", "US"):
        m = pc == cn
        pcor[m] = correct(p0[m], seg[m], pri[cn], label=f"[val {cn}] ")
    print(f"unshifted val: plain {f05(V, p0)}  corrected {f05(V, pcor)}")

    # simulation: 2 extra competing negatives per entity that has one, p drawn
    # from the real competing-negative pool of the same country
    rng = np.random.default_rng(0)
    add_g, add_p = [], []
    for cn in ("India", "US"):
        neg = np.flatnonzero((seg == 1) & ~V["y"] & (V["p"] >= MIN_P) & (pc == cn))
        ents = np.unique(V["g"][(seg == 1) & ~V["y"] & (pc == cn)])
        add_g.append(np.repeat(ents, 2))
        add_p.append(V["p"][rng.choice(neg, 2 * len(ents))])
    add_g = np.concatenate(add_g); add_p = np.concatenate(add_p)
    base = int(V["tid"].max()) + 1
    V2 = dict(V)
    V2["g"] = np.concatenate([V["g"], add_g])
    V2["p"] = np.concatenate([V["p"], add_p])
    V2["y"] = np.concatenate([V["y"], np.zeros(len(add_g), bool)])
    V2["src"] = np.concatenate([V["src"], np.full(len(add_g), 2, V["src"].dtype)])
    V2["tid"] = np.concatenate([V["tid"], base + np.arange(len(add_g)).astype(V["tid"].dtype)])
    seg2 = np.concatenate([seg, np.ones(len(add_g), np.int8)])
    pc2 = V["ctry"][V2["g"]]
    print(f"\nsimulation: injected {len(add_g):,} decoys")
    pcor2 = V2["p"].copy()
    for cn in ("India", "US"):
        m = pc2 == cn
        pcor2[m] = correct(V2["p"][m], seg2[m], pri[cn], label=f"[sim {cn}] ")
    print(f"shifted val:   plain {f05(V2, V2['p'])}  corrected {f05(V2, pcor2)}")


def count_prior(n_pairs_ent_test, true_per_ent):
    """Second estimator: true matches per entity in a segment are a property of
    the noise process (same in train and test); decoys only add pairs."""
    return float(np.clip(true_per_ent / max(n_pairs_ent_test, 1e-9), 1e-3, 0.999))


def main_test():
    V, vseg, _, pc, pri = val_segments()
    n_ent = {cn: int((V["ctry"] == cn).sum()) for cn in ("India", "US")}
    true_pe = {cn: {g: float((V["y"] & (vseg == g) & (V["p"] >= MIN_P) & (pc == cn)).sum())
                    / n_ent[cn] for g in range(NSEG)} for cn in ("India", "US")}
    true_pe["France"] = {g: 0.5 * (true_pe["India"][g] + true_pe["US"][g]) for g in range(NSEG)}
    meta = pl.read_parquet(f"{CACHE}/test/s1/meta.parquet", columns=["country"])
    for f, cn in (("fr", "France"), ("us", "US"), ("in", "India")):
        d = dict(np.load(f"{ROOT}/work/dumps/{f}.npz"))
        keys, segs = seg_index("test", [cn])
        seg, ok = lookup(keys, segs, d["row"], d["src"], d["tid"])
        ne = int((meta["country"] == cn).sum())
        p = d["p"].astype(np.float64)
        q = p.copy()
        print(f"{f}: {len(seg):,} scored pairs, {ok.mean() * 100:.2f}% segmented, {ne:,} entities")
        for g in range(NSEG):
            m = (seg == g) & (p >= MIN_P)
            if m.sum() < 100:
                continue
            pi_tr = pri[cn][g][0]
            pi_em = em_prior(p[m], pi_tr)
            pi_ct = count_prior(m.sum() / ne, true_pe[cn][g])
            # France is out of distribution for EM's stable-likelihood
            # assumption; use the count estimator there, EM elsewhere
            pi_te = pi_ct if cn == "France" else pi_em
            q[m] = adjust(p[m], pi_tr, pi_te)
            print(f"   [{cn}] {SEG_NAMES[g]:<10} n/ent {m.sum() / ne:.3f}  ref {pi_tr:.3f}  "
                  f"EM {pi_em:.3f}  count {pi_ct:.3f}  -> used {pi_te:.3f}")
        d["p"] = q.astype(np.float32)
        np.savez(f"{ROOT}/work/dumps/{f}_sf.npz", **d)


if __name__ == "__main__":
    main_val() if sys.argv[1] == "val" else main_test()
