"""
Error taxonomy on the validation fold.

Buckets every mistake by a structural property of the pair (empty address,
cross-script name, name token substituted, weak blocking evidence, contested
target), so the next iteration is driven by where the loss actually is rather
than by guesswork.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
MODELS = f"{ROOT}/work/models"
CTAG = os.environ.get("CTAG", "c1")
VAL_FOLD = int(os.environ.get("VAL_FOLD", "0"))
THR = float(os.environ.get("THR", "0.8"))


def main():
    z = np.load(f"{MODELS}/val_scores_{CTAG}.npz")
    gt, p, pos, src, tid = z["gt"], z["p"], z["pos"], z["src"], z["tid"]
    T, fold, g2gt = z["T"], z["fold"], z["g2gt"]
    inv = np.empty(len(T), np.int64); inv[g2gt] = np.arange(len(g2gt))

    m1 = pl.read_parquet(f"{CACHE}/train/s1/meta.parquet")
    mt = {s: pl.read_parquet(f"{CACHE}/train/s{s}/meta.parquet") for s in (2, 3)}
    idx_t = {}
    for s in (2, 3):
        ids = mt[s]["id"].to_numpy()
        o = np.argsort(ids)
        idx_t[s] = (ids[o], o)

    s1rowv = inv[gt]
    def trow(s, t):
        ids, o = idx_t[s]
        j = np.searchsorted(ids, t)
        j = np.clip(j, 0, len(ids) - 1)
        return np.where(ids[j] == t, o[j], -1)
    tr = np.empty(len(tid), np.int64)
    for s in (2, 3):
        m = src == s
        tr[m] = trow(s, tid[m])

    # ---- structural properties of each pair
    a1 = m1["addr_norm"].to_numpy()
    n1 = m1["name_toks"].to_numpy()
    q_noaddr = np.array([len(x) == 0 for x in a1])[s1rowv]
    c_noaddr = np.zeros(len(tid), bool)
    c_indic = np.zeros(len(tid), bool)
    c_nt = np.empty(len(tid), dtype=object)
    for s in (2, 3):
        m = src == s
        aa = mt[s]["addr_norm"].to_numpy()
        nn = mt[s]["name_toks"].to_numpy()
        raw = mt[s]["name_norm"].to_numpy()
        c_noaddr[m] = np.array([len(aa[r]) == 0 for r in tr[m]])
        c_nt[m] = np.array([nn[r] for r in tr[m]], dtype=object)
    q_nt = n1[s1rowv]
    shared_name = np.array([len(set(a.split("\t")) & set(b.split("\t"))) > 0
                            if a and b else False
                            for a, b in zip(q_nt, c_nt)])

    fn = (pos == 1) & (p < THR)          # true match the model rejected
    fp = (pos == 0) & (p >= THR)         # non-match the model accepted
    print(f"validation pairs {len(p):,}   true {int(pos.sum()):,}")
    print(f"at threshold {THR}: FN {int(fn.sum()):,} "
          f"({fn.sum()/max(pos.sum(),1)*100:.2f}% of true)   "
          f"FP {int(fp.sum()):,} "
          f"({fp.sum()/max((p>=THR).sum(),1)*100:.2f}% of accepted)")

    def bucket(mask, label):
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<40} 0")
            return
        print(f"  {label:<40} {n:>9,}  "
              f"noaddr(cand) {c_noaddr[mask].mean()*100:5.1f}%  "
              f"noaddr(S1) {q_noaddr[mask].mean()*100:5.1f}%  "
              f"share-name-tok {shared_name[mask].mean()*100:5.1f}%  "
              f"mean p {p[mask].mean():.3f}")

    print("\nFALSE NEGATIVES (true matches scored below threshold)")
    bucket(fn, "all FN")
    bucket(fn & c_noaddr, "  candidate has no address")
    bucket(fn & ~c_noaddr, "  candidate has an address")
    bucket(fn & ~shared_name, "  no shared name token")
    bucket(fn & shared_name, "  shares a name token")
    print("\nFALSE POSITIVES (non-matches scored above threshold)")
    bucket(fp, "all FP")
    bucket(fp & c_noaddr, "  candidate has no address")
    bucket(fp & shared_name, "  shares a name token")

    print("\n--- sample FALSE NEGATIVES (highest-T entities first) ---")
    show(fn, p, s1rowv, tr, src, m1, mt, 10)
    print("\n--- sample FALSE POSITIVES (highest p first) ---")
    show(fp, p, s1rowv, tr, src, m1, mt, 10, by_p=True)


def show(mask, p, s1rowv, tr, src, m1, mt, k, by_p=False):
    idx = mask.nonzero()[0]
    if len(idx) == 0:
        return
    order = np.argsort(-p[idx]) if by_p else np.argsort(p[idx])
    for i in idx[order[:k]]:
        r1 = m1.row(int(s1rowv[i]), named=True)
        r2 = mt[int(src[i])].row(int(tr[i]), named=True)
        print(f"  p={p[i]:.4f}")
        print(f"    S1 {r1['name_norm']!r}\n       {r1['addr_norm']!r}")
        print(f"    S{src[i]} {r2['name_norm']!r}\n       {r2['addr_norm']!r}")


if __name__ == "__main__":
    main()
