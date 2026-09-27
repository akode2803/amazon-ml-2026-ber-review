"""
Qualitative + quantitative look at the two big false-negative families on the
validation fold:

  MISSED   true pair never retrieved by blocking
  REJECTED true pair retrieved but scored below the decision (p < P_REJ)

For each family: a random sample of raw (S1, target) normalized records, and
a coarse taxonomy computed from the cached token fields:

  name_share   any name token/skeleton in common
  addr_share   any address token/skeleton in common
  num_share    any building/street number in common

Usage: python diag_fn.py [ctag] [n_print]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
MODELS = f"{ROOT}/work/models"
P_REJ = 0.5


def toks(s):
    return set(x for x in s.split("\t") if x) if s else set()


def taxonomy(a, b):
    ns = bool(toks(a["name_toks"]) & toks(b["name_toks"]) or
              toks(a["name_skels"]) & toks(b["name_skels"]))
    ad = bool(toks(a["addr_toks"]) & toks(b["addr_toks"]) or
              toks(a["addr_skels"]) & toks(b["addr_skels"]))
    nu = bool(toks(a["nums"]) & toks(b["nums"]))
    return ns, ad, nu


def main():
    ctag = sys.argv[1] if len(sys.argv) > 1 else "c1"
    n_print = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    rng = np.random.default_rng(3)
    cols = ["id", "country", "name_norm", "addr_norm", "name_toks", "addr_toks",
            "nums", "name_skels", "addr_skels"]
    s1 = pl.read_parquet(f"{CACHE}/train/s1/meta.parquet", columns=cols)
    tg = {s: pl.read_parquet(f"{CACHE}/train/s{s}/meta.parquet", columns=cols)
          for s in (2, 3)}
    s1row, pq, ps, tgt_row = M.gt_arrays("train")
    fold = M.fold_of(s1["id"].to_numpy())
    ctry1 = s1["country"].to_numpy()

    # ---------------- MISSED: gold pairs of val entities absent from cands
    missed = []
    for country in ("India", "US"):
        for source in (2, 3):
            z = np.load(f"{CACHE}/train/cand/{ctag}_{country}_s{source}.npz")
            rows1, rowsT, q, c = z["rows1"], z["rowsT"], z["q"], z["c"]
            have = np.sort(rows1[q].astype(np.int64) * (1 << 24) + rowsT[c])
            sel = ps == source
            r1 = s1row[pq[sel]]
            rt = tgt_row[sel]
            ok = (ctry1[r1] == country) & (fold[r1] == 0)
            key = r1[ok].astype(np.int64) * (1 << 24) + rt[ok]
            j = np.clip(np.searchsorted(have, key), 0, len(have) - 1)
            miss = have[j] != key
            for a, b in zip(r1[ok][miss], rt[ok][miss]):
                missed.append((country, source, int(a), int(b)))
    print(f"MISSED val true pairs: {len(missed):,}")

    # ---------------- REJECTED: from the val score file
    z = np.load(f"{MODELS}/val_scores_{ctag}.npz")
    gt, p, pos, src, tid = z["gt"], z["p"], z["pos"], z["src"], z["tid"]
    g2gt = z["g2gt"]
    inv = np.full(int(g2gt.max()) + 2, -1, np.int64)
    ok = g2gt >= 0
    inv[g2gt[ok]] = np.nonzero(ok)[0]
    rej = np.nonzero(pos & (p < P_REJ))[0]
    id2row = {s: dict(zip(tg[s]["id"].to_list(), range(tg[s].height)))
              for s in (2, 3)}
    rejected = [(str(ctry1[int(inv[gt[i]])]), int(src[i]),
                 int(inv[gt[i]]), id2row[int(src[i])][int(tid[i])], float(p[i]))
                for i in rng.choice(rej, min(20000, len(rej)), replace=False)]
    print(f"REJECTED val true pairs (p<{P_REJ}): {len(rej):,}  "
          f"p-hist {np.histogram(p[rej], bins=[0, .02, .1, .2, .3, .4, .5])[0]}")

    for label, fam in (("MISSED", missed), ("REJECTED", rejected)):
        cnt = {}
        for row in fam[:20000]:
            a = s1.row(row[2], named=True)
            b = tg[row[1]].row(row[3], named=True)
            k = taxonomy(a, b)
            cnt[k] = cnt.get(k, 0) + 1
        tot = sum(cnt.values())
        print(f"\n== {label} taxonomy (name_share, addr_share, num_share) of {tot:,}")
        for k, v in sorted(cnt.items(), key=lambda t: -t[1]):
            print(f"   {str(k):<24} {v:>7,} {v / tot * 100:6.2f}%")
        by_c = {}
        for row in fam:
            by_c[(row[0], row[1])] = by_c.get((row[0], row[1]), 0) + 1
        print("   by country/source:", by_c)
        print(f"\n== {label} examples")
        for i in rng.choice(len(fam), min(n_print, len(fam)), replace=False):
            row = fam[i]
            a = s1.row(row[2], named=True)
            b = tg[row[1]].row(row[3], named=True)
            extra = f" p={row[4]:.3f}" if len(row) > 4 else ""
            print(f" [{row[0]} S{row[1]}{extra}]\n   S1: {a['name_norm']!r:.60} | {a['addr_norm']!r:.90}"
                  f"\n   T : {b['name_norm']!r:.60} | {b['addr_norm']!r:.90}")


if __name__ == "__main__":
    main()
