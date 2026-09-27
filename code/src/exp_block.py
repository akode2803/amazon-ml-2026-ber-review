"""
Experiment: measure candidate-generation quality on train.

Reports, per (country, source) partition and overall:
    candidate recall          fraction of true matches present in the candidate set
    entity-level recall       fraction of S1 entities whose *every* match was found
    cands/S1                  mean / median / p95 / p99 / max
    total candidate pairs
Also reports country agreement between an S1 entity and its true matches, which
decides whether country is a safe hard block.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
DATA = f"{ROOT}/student_resource/dataset"


def id_lookup(split, source):
    """Return (sorted_ids, order) so a numeric entity id -> global row index."""
    ids = pl.read_parquet(f"{CACHE}/{split}/s{source}/meta.parquet",
                          columns=["id"])["id"].to_numpy()
    order = np.argsort(ids).astype(np.int32)
    return ids[order], order


def lookup(sorted_ids, order, q):
    pos = np.searchsorted(sorted_ids, q)
    pos = np.clip(pos, 0, len(sorted_ids) - 1)
    hit = sorted_ids[pos] == q
    return np.where(hit, order[pos], -1)


def load_gt(split="train"):
    """GT as flat arrays: s1row, tgt_source, tgt_row (global row indices)."""
    gt = pl.read_csv(f"{DATA}/{split}/{split}_ground_truth.tsv", separator="\t",
                     quote_char=None,
                     schema_overrides={"source1_entity_id": pl.Utf8,
                                       "matched_entity_ids": pl.Utf8})
    s1num = gt["source1_entity_id"].str.slice(3).cast(pl.Int32).to_numpy()
    pair_q, pair_s, pair_t = [], [], []
    for qi, rest in enumerate(gt["matched_entity_ids"].fill_null("").to_list()):
        if not rest:
            continue
        for m in rest.split(","):
            pair_q.append(qi)
            pair_s.append(2 if m[1] == "2" else 3)
            pair_t.append(int(m[3:]))
    return (s1num, np.asarray(pair_q, np.int32), np.asarray(pair_s, np.int8),
            np.asarray(pair_t, np.int32))


def main():
    t0 = time.time()
    split = "train"
    s1num, pq, ps, pt = load_gt(split)
    print(f"GT: {len(s1num):,} S1 rows, {len(pq):,} true pairs "
          f"({time.time()-t0:.0f}s)")

    # -- map GT ids to row indices
    s1_sorted, s1_order = id_lookup(split, 1)
    s1row = lookup(s1_sorted, s1_order, s1num)
    assert (s1row >= 0).all(), "GT references an S1 id not in source1"
    meta1 = pl.read_parquet(f"{CACHE}/{split}/s1/meta.parquet", columns=["country"])
    c1 = meta1["country"].to_numpy()

    tgt_row = np.full(len(pq), -1, np.int32)
    tgt_country = np.empty(len(pq), dtype=object)
    for s in (2, 3):
        sel = ps == s
        ss, so = id_lookup(split, s)
        tgt_row[sel] = lookup(ss, so, pt[sel])
        cc = pl.read_parquet(f"{CACHE}/{split}/s{s}/meta.parquet",
                             columns=["country"])["country"].to_numpy()
        tgt_country[sel] = cc[tgt_row[sel]]
        del cc
    assert (tgt_row >= 0).all(), "GT references a target id not in source2/3"

    # -- country agreement
    q_country = c1[s1row][pq]
    agree = (q_country == tgt_country)
    print(f"\ncountry agreement on true pairs: {agree.mean()*100:.4f}% "
          f"({(~agree).sum():,} disagreements of {len(agree):,})")
    if (~agree).sum():
        bad = pl.DataFrame({"s1": q_country[~agree], "tgt": tgt_country[~agree]})
        print(bad.group_by(["s1", "tgt"]).len().sort("len", descending=True).head(10))

    np.savez(f"{CACHE}/gt_{split}.npz", s1num=s1num, s1row=s1row, pq=pq, ps=ps,
             pt=pt, tgt_row=tgt_row)
    print(f"\nsaved GT row maps ({time.time()-t0:.0f}s)")
    if sys.argv[1:2] == ["gt"]:          # pipeline step: GT maps only
        return

    # -- blocking sweep
    countries = sorted(set(c1.tolist()))
    DF_MAX, BUDGET = 3000, 4000
    for topk in [int(x) for x in (sys.argv[1:] or ["20"])]:
        print(f"\n{'='*84}\nTOPK={topk}  DF_MAX={DF_MAX}  BUDGET={BUDGET}\n{'='*84}")
        tot_found = tot_true = tot_cand = 0
        per_entity = np.zeros(len(s1row), np.int32)
        found_per_entity = np.zeros(len(s1row), np.int32)
        true_per_entity = np.zeros(len(s1row), np.int32)
        for c in countries:
            qsel = (c1[s1row] == c).nonzero()[0]         # GT row indices
            for s in (2, 3):
                tb = time.time()
                keep, tptr, tval, idx = block.build_index(split, s, c)
                gl2loc = np.full(
                    int(pl.read_parquet(f"{CACHE}/{split}/s{s}/meta.parquet",
                                        columns=["id"]).height), -1, np.int32)
                gl2loc[keep] = np.arange(len(keep), dtype=np.int32)

                # queries: the S1 rows of this country, in GT order
                q_global = s1row[qsel]
                z = np.load(f"{CACHE}/{split}/s1/terms.npz")
                a_ptr, a_val = z["tptr"], z["tval"]
                lens = (a_ptr[q_global + 1] - a_ptr[q_global]).astype(np.int64)
                q_ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
                q_val = np.empty(q_ptr[-1], np.uint32)
                block._gather_csr(a_ptr, a_val, q_global.astype(np.int64), q_ptr, q_val)
                del z, a_ptr, a_val
                q_norm = block.query_norms(q_ptr, q_val, idx)
                tq = time.time()
                cand, score, cnt = block.query_index(
                    idx, q_ptr, q_val, q_norm, df_max=DF_MAX, budget=BUDGET,
                    topk=topk)
                qtime = time.time() - tq

                # -- recall for this (country, source)
                mask = (ps == s) & np.isin(pq, qsel)
                sub_pq, sub_tr = pq[mask], tgt_row[mask]
                pos_in_q = np.full(len(s1row), -1, np.int64)
                pos_in_q[qsel] = np.arange(len(qsel))
                rows = pos_in_q[sub_pq]
                loc = gl2loc[sub_tr]
                hit = np.zeros(len(rows), bool)
                for a in range(topk):
                    hit |= (cand[rows, a] == loc) & (loc >= 0)
                nc = cnt.astype(np.int64)
                print(f"  {c:7s} S{s}: idx {len(keep):>9,} rec / "
                      f"{len(idx.vocab):>9,} terms / {idx.bytes()/1e6:6.0f} MB | "
                      f"q {len(qsel):>9,} in {qtime:5.1f}s | "
                      f"recall {hit.mean()*100:6.2f}% ({hit.sum():,}/{len(hit):,}) | "
                      f"cands/S1 {nc.mean():5.1f} p95 {np.percentile(nc,95):.0f} | "
                      f"build {tq-tb:.0f}s", flush=True)
                tot_found += hit.sum(); tot_true += len(hit); tot_cand += nc.sum()
                per_entity[qsel] += cnt
                np.add.at(found_per_entity, sub_pq, hit.astype(np.int32))
                np.add.at(true_per_entity, sub_pq, 1)
                del idx, keep, tptr, tval, cand, score, cnt, gl2loc
        ent_ok = (found_per_entity == true_per_entity)
        print(f"\n  OVERALL  pair recall {tot_found/tot_true*100:.3f}%  "
              f"({tot_found:,}/{tot_true:,})")
        print(f"           entities with ALL matches found: {ent_ok.mean()*100:.2f}%")
        print(f"           total candidate pairs {tot_cand:,}  "
              f"mean/S1 {per_entity.mean():.1f}  median {np.median(per_entity):.0f}  "
              f"p95 {np.percentile(per_entity,95):.0f}  p99 {np.percentile(per_entity,99):.0f}  "
              f"max {per_entity.max()}")
        print(f"           S1 with 0 candidates: {(per_entity==0).mean()*100:.2f}%")
        print(f"  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
