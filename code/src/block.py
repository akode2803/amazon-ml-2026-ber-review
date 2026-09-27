"""
Stage 3: candidate generation -- budgeted, IDF-weighted rare-term retrieval
over an inverted index, partitioned by (country, source).

The design point that matters: instead of a fixed blocking key or a full
TF-IDF cosine against the whole corpus, each S1 query walks *its own rarest
terms first* and stops once a posting budget is spent.  That gives

  * a hard O(N_query x BUDGET) bound on work, independent of how common a
    record's tokens happen to be (no "Delhi" blow-up), and
  * candidates ranked by the evidence that actually discriminates -- rare term
    agreement -- rather than by raw string overlap.

Scoring is  sum(idf^2 over shared terms) / (|q|^aq * |c|^ac).  aq is constant
within a query so it cannot change the ranking, only make a global score
threshold comparable across queries; ac is a real ranking knob and is swept.
"""
from __future__ import annotations

import os

import numba
import numpy as np
import polars as pl

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"


@numba.njit(cache=True)
def _scatter_postings(term_id, rec_of, ptr, post):
    cur = ptr[:-1].copy()
    for i in range(term_id.shape[0]):
        t = term_id[i]
        post[cur[t]] = rec_of[i]
        cur[t] += 1


def load_terms(split, source, country, tag="v2"):
    """Return (global_rows, csr_ptr, csr_val) for one (country, source) block."""
    z = np.load(f"{CACHE}/{split}/s{source}/terms_{tag}_{country}.npz")
    return z["rows"], z["tptr"], z["tval"]


class Index:
    """Inverted index over one (country, source) partition of target records."""

    def __init__(self, ptr, val, n_rec):
        self.n = n_rec
        uniq, inv = np.unique(val, return_inverse=True)
        inv = inv.astype(np.int32)
        self.vocab = uniq
        df = np.bincount(inv, minlength=len(uniq)).astype(np.int32)
        self.df = df
        self.ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(df)])
        self.post = np.empty(self.ptr[-1], dtype=np.int32)
        rec_of = np.repeat(np.arange(n_rec, dtype=np.int32),
                           (ptr[1:] - ptr[:-1]).astype(np.int32))
        _scatter_postings(inv, rec_of, self.ptr, self.post)
        self.idf = np.log(1.0 + n_rec / np.maximum(df, 1)).astype(np.float32)
        w = self.idf[inv].astype(np.float64) ** 2
        cs = np.concatenate([np.zeros(1, np.float64), np.cumsum(w)])
        self.norm = np.maximum(np.sqrt(cs[ptr[1:]] - cs[ptr[:-1]]), 1e-6).astype(np.float32)

    def nbytes(self):
        return (self.vocab.nbytes + self.df.nbytes + self.ptr.nbytes
                + self.post.nbytes + self.idf.nbytes + self.norm.nbytes)


@numba.njit(parallel=True, cache=True)
def _query(q_ptr, q_term, q_norm,
           i_ptr, i_post, i_df, i_idf, i_norm,
           df_max, budget, topk, aq, ac, min_score,
           out_cand, out_score, out_cnt, out_touch, scratch, touched):
    nthread = scratch.shape[0]
    cap = touched.shape[1]
    for qi in numba.prange(q_ptr.shape[0] - 1):
        tid = numba.get_thread_id() % nthread
        sc = scratch[tid]
        tch = touched[tid]
        lo, hi = q_ptr[qi], q_ptr[qi + 1]
        m = hi - lo
        if m == 0:
            out_cnt[qi] = 0
            out_touch[qi] = 0
            continue
        dfs = np.empty(m, dtype=np.int64)
        for j in range(m):
            t = q_term[lo + j]
            dfs[j] = i_df[t] if t >= 0 else (1 << 40)
        order = np.argsort(dfs)

        ntouch = 0
        spent = 0
        full = False
        for oi in range(m):
            if full:
                break
            t = q_term[lo + order[oi]]
            if t < 0:
                break
            d = i_df[t]
            if d > df_max:
                break                       # remaining terms are only commoner
            if spent + d > budget and ntouch > 0:
                break                       # posting budget exhausted
            w = i_idf[t] * i_idf[t]
            for p in range(i_ptr[t], i_ptr[t + 1]):
                r = i_post[p]
                if sc[r] == 0.0:
                    if ntouch >= cap:       # scratch full: stop cleanly so that
                        full = True         # every touched record stays resettable
                        break
                    tch[ntouch] = r
                    ntouch += 1
                sc[r] += w
            spent += d

        out_touch[qi] = ntouch
        nkeep = 0
        if ntouch > 0:
            qn = q_norm[qi] ** aq
            # bounded top-k by insertion: O(ntouch * k) worst case but with an
            # early-exit on the running k-th best it is ~O(ntouch) in practice,
            # and it allocates nothing.  A full argsort over the ~2.3k touched
            # records dominated runtime before.
            bs = out_score[qi]
            bc = out_cand[qi]
            worst = min_score
            for a in range(ntouch):
                r = tch[a]
                v = sc[r] / (qn * i_norm[r] ** ac)
                if v < worst:
                    continue
                if nkeep < topk:
                    j = nkeep
                    nkeep += 1
                else:
                    j = topk - 1
                    if v <= bs[j]:
                        continue
                while j > 0 and bs[j - 1] < v:
                    bs[j] = bs[j - 1]
                    bc[j] = bc[j - 1]
                    j -= 1
                bs[j] = v
                bc[j] = r
                if nkeep == topk:
                    worst = bs[topk - 1]
                    if worst < min_score:
                        worst = min_score
        out_cnt[qi] = nkeep
        for a in range(ntouch):
            sc[tch[a]] = 0.0


def map_terms(idx, q_val):
    t = np.searchsorted(idx.vocab, q_val)
    t = np.clip(t, 0, len(idx.vocab) - 1).astype(np.int64)
    t[idx.vocab[t] != q_val] = -1
    return t


def query_norms(ptr, val, idx):
    """L2 norm of a query record measured with the *target* partition's idfs."""
    t = np.searchsorted(idx.vocab, val)
    t = np.clip(t, 0, len(idx.vocab) - 1)
    w = np.where(idx.vocab[t] == val, idx.idf[t], 0.0).astype(np.float64) ** 2
    cs = np.concatenate([np.zeros(1, np.float64), np.cumsum(w)])
    return np.maximum(np.sqrt(cs[ptr[1:]] - cs[ptr[:-1]]), 1e-6).astype(np.float32)


def query_index(idx, q_ptr, q_val, df_max=3000, budget=4000, topk=30,
                aq=1.0, ac=1.0, min_score=0.0, max_touch=60000, nthread=None):
    # one scratch row per numba worker: a shared row would race silently
    nthread = nthread or numba.get_num_threads()
    q_term = map_terms(idx, q_val)
    q_norm = query_norms(q_ptr, q_val, idx)
    nq = len(q_ptr) - 1
    out_cand = np.full((nq, topk), -1, dtype=np.int32)
    out_score = np.zeros((nq, topk), dtype=np.float32)
    out_cnt = np.zeros(nq, dtype=np.int32)
    out_touch = np.zeros(nq, dtype=np.int32)
    scratch = np.zeros((nthread, idx.n), dtype=np.float32)
    touched = np.zeros((nthread, max_touch), dtype=np.int32)
    _query(q_ptr, q_term, q_norm, idx.ptr, idx.post, idx.df, idx.idf, idx.norm,
           df_max, budget, topk, aq, ac, min_score,
           out_cand, out_score, out_cnt, out_touch, scratch, touched)
    del scratch, touched
    return out_cand, out_score, out_cnt, out_touch


def subset_csr(rows_wanted, rows_all, ptr, val):
    """Re-slice a CSR keyed by global row ids down to `rows_wanted` order."""
    pos = np.full(int(rows_all.max()) + 2, -1, np.int32)
    pos[rows_all] = np.arange(len(rows_all), dtype=np.int32)
    loc = pos[rows_wanted].astype(np.int64)
    lens = (ptr[loc + 1] - ptr[loc]).astype(np.int64)
    new_ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
    out = np.empty(new_ptr[-1], dtype=np.uint32)
    _gather(ptr, val, loc, new_ptr, out)
    return new_ptr, out


@numba.njit(cache=True)
def _gather(ptr, val, loc, new_ptr, out):
    for i in range(loc.shape[0]):
        r = loc[i]
        o = new_ptr[i]
        for j in range(ptr[r], ptr[r + 1]):
            out[o] = val[j]
            o += 1
