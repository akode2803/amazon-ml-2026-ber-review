"""
Shared plumbing between candidate generation and the matching model:
context features, gold labels, and chunked feature computation.

The context block is where the *competition* signals live.  Two of them do
most of the precision work:

  blk_margin / p-margin  -- how far ahead of the runner-up this candidate is
                            for its S1 entity.  A 0.8 candidate with a 0.79
                            rival is a coin flip; a 0.8 with no rival is not.
  tgt_rank / tgt_rel     -- the ground truth is many-to-one: every S2/S3 record
                            in train matches **exactly one** S1 entity
                            (verified: 7,638,365 references, 7,638,365 distinct
                            targets).  So a candidate that some *other* S1
                            entity wants more is probably that entity's.
"""
from __future__ import annotations

import numpy as np


def group_stats(key, val):
    """Per-group rank (by val desc), group max, runner-up value and size."""
    n = len(key)
    if n == 0:
        z = np.zeros(0, np.float32)
        return (np.zeros(0, np.int32), z, z, np.zeros(0, np.int32))
    o = np.lexsort((-val, key))
    k, v = key[o], val[o]
    first = np.ones(n, bool)
    first[1:] = k[1:] != k[:-1]
    starts = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    rank = (np.arange(n) - starts[gid]).astype(np.int32)
    gmax = v[starts][gid]
    cnt = np.diff(np.append(starts, n))
    second = np.where(cnt > 1, v[np.minimum(starts + 1, n - 1)], 0.0)[gid]
    size = cnt[gid].astype(np.int32)
    inv = np.empty(n, np.int64)
    inv[o] = np.arange(n)
    return rank[inv], gmax[inv], second[inv], size[inv]


def gold_mask(q, c, nT, gq, gc):
    """Boolean: is candidate pair (q,c) in the gold set (gq,gc)?"""
    if len(gq) == 0 or len(q) == 0:
        return np.zeros(len(q), bool)
    gold = np.sort(gq.astype(np.int64) * nT + gc.astype(np.int64))
    key = q.astype(np.int64) * nT + c.astype(np.int64)
    j = np.searchsorted(gold, key)
    j = np.clip(j, 0, len(gold) - 1)
    return gold[j] == key


def context(q, c, s, source, qstore, cstore, frank=None, rrank=None):
    """The context columns, in features.CTX_NAMES order.

    fwd_rank / rev_rank are the raw retrieval ranks from the two directions.
    They are not redundant with blk_rank: rev_rank says "this candidate record
    ranks *this* S1 entity Nth among all entities it could belong to", which is
    the many-to-one constraint expressed from the candidate's own side.
    """
    n = len(q)
    ctx = np.zeros((n, 16), np.float32)
    qrank, qmax, qsecond, qsize = group_stats(q, s)
    crank, cmax, _, _ = group_stats(c, s)
    ctx[:, 0] = s
    ctx[:, 1] = qrank
    ctx[:, 2] = s / np.maximum(qmax, 1e-9)
    ctx[:, 3] = s - qsecond
    ctx[:, 4] = qsize
    ctx[:, 5] = cmax
    ctx[:, 6] = s / np.maximum(cmax, 1e-9)
    ctx[:, 7] = crank
    ctx[:, 8] = 1.0 if source == 3 else 0.0
    qa = qstore.has_addr[q]
    ca = cstore.has_addr[c]
    ctx[:, 9] = qa
    ctx[:, 10] = ca
    ctx[:, 11] = qa * ca
    ln_q = np.array([len(x) for x in qstore.name_norm], np.float32)[q]
    ln_c = np.array([len(x) for x in cstore.name_norm], np.float32)[c]
    la_q = np.array([len(x) for x in qstore.addr_norm], np.float32)[q]
    la_c = np.array([len(x) for x in cstore.addr_norm], np.float32)[c]
    ctx[:, 12] = np.minimum(ln_q, ln_c) / np.maximum(np.maximum(ln_q, ln_c), 1)
    ctx[:, 13] = np.minimum(la_q, la_c) / np.maximum(np.maximum(la_q, la_c), 1)
    CAP = 60.0          # NORANK (9999) would dominate the split thresholds
    ctx[:, 14] = np.minimum(frank, CAP) if frank is not None else CAP
    ctx[:, 15] = np.minimum(rrank, CAP) if rrank is not None else CAP
    return ctx
