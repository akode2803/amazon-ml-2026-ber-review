"""
Recovery-block experiment: how many pairs that retrieval *missed* share an
exact / loose address signature or an exact name signature with their S1,
and what does a signature block cost in extra candidates?

A signature block joins S1 and target records on a whole-field hash, capped
to low-multiplicity (= high-information) signatures, so it adds very few
pairs by construction.  Only pairs not already in the candidate set count.

Usage: python recovery.py [ctag]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
import pairs as P  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"


def sig_join(hq, hc, cap_q, cap_c):
    """All (i, j) with hq[i] == hc[j] != 0, for keys held by <= cap_q S1 rows
    and <= cap_c target rows.  Returns (qi, ci) int64 arrays."""
    uq, cq = np.unique(hq[hq != 0], return_counts=True)
    uc, cc = np.unique(hc[hc != 0], return_counts=True)
    ok_q = uq[cq <= cap_q]
    ok_c = uc[cc <= cap_c]
    keys = np.intersect1d(ok_q, ok_c)
    if len(keys) == 0:
        z = np.zeros(0, np.int64)
        return z, z
    qi = np.flatnonzero(np.isin(hq, keys))
    oc = np.argsort(hc, kind="stable")
    hcs = hc[oc]
    lo = np.searchsorted(hcs, hq[qi], "left")
    hi = np.searchsorted(hcs, hq[qi], "right")
    n = hi - lo
    rq = np.repeat(qi, n)
    start = np.repeat(lo, n)
    within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    rc = oc[start + within]
    return rq.astype(np.int64), rc.astype(np.int64)


def main():
    ctag = sys.argv[1] if len(sys.argv) > 1 else "c1"
    split = "train"
    s1row, pq, ps, tgt_row = M.gt_arrays(split)
    tot = {}
    n_true_all = n_miss_all = 0
    n_s1 = 0
    for country in M.countries_of(split):
        s1sig = np.load(f"{CACHE}/{split}/sig_{country}_s1.npz")
        for source in (2, 3):
            z = np.load(f"{CACHE}/{split}/cand/{ctag}_{country}_s{source}.npz")
            rows1, rowsT, q, c = z["rows1"], z["rowsT"], z["q"], z["c"]
            n1, nT = len(rows1), len(rowsT)
            if source == 2:
                n_s1 += n1
            tsig = np.load(f"{CACHE}/{split}/sig_{country}_s{source}.npz")
            # gold pairs of this partition in local coordinates
            qpos = np.full(int(max(rows1.max(), s1row.max())) + 2, -1, np.int64)
            qpos[rows1] = np.arange(n1)
            locT = np.full(int(rowsT.max()) + 2, -1, np.int64)
            locT[rowsT] = np.arange(nT)
            sel = ps == source
            gq = qpos[s1row[pq[sel]]]
            gc = locT[np.minimum(tgt_row[sel], len(locT) - 1)]
            ok = (gq >= 0) & (gc >= 0)
            gq, gc = gq[ok], gc[ok]
            have = np.sort(q.astype(np.int64) * nT + c)
            gkey = gq * nT + gc
            found = P.gold_mask(q, c, nT, gq, gc)
            j = np.clip(np.searchsorted(have, gkey), 0, len(have) - 1)
            missed = have[j] != gkey
            miss_key = np.sort(gkey[missed])
            n_true_all += len(gkey)
            n_miss_all += int(missed.sum())
            for sig in ("ad", "adl", "nm"):
                for cap in (1, 2, 3, 5, 10):
                    rq, rc = sig_join(s1sig[sig], tsig[sig], cap, cap * 3)
                    k = rq * nT + rc
                    jj = np.clip(np.searchsorted(have, k), 0, len(have) - 1)
                    new = have[jj] != k
                    k = k[new]
                    jm = np.clip(np.searchsorted(miss_key, k), 0,
                                 max(len(miss_key) - 1, 0))
                    rec = int((miss_key[jm] == k).sum()) if len(miss_key) else 0
                    a = tot.setdefault((sig, cap), [0, 0])
                    a[0] += len(k); a[1] += rec
            del found
    print(f"true pairs {n_true_all:,}  missed by retrieval {n_miss_all:,} "
          f"({n_miss_all / n_true_all * 100:.3f}%)  S1 {n_s1:,}")
    print(f"\n  {'signature':<8} {'cap':>4} {'new pairs':>11} {'/S1':>7} "
          f"{'recovered':>10} {'% of miss':>9} {'precision':>9}")
    for (sig, cap), (nn, rr) in sorted(tot.items()):
        print(f"  {sig:<8} {cap:>4} {nn:>11,} {nn / n_s1:>7.3f} {rr:>10,} "
              f"{rr / max(n_miss_all, 1) * 100:>8.2f}% {rr / max(nn, 1) * 100:>8.2f}%")


if __name__ == "__main__":
    main()
