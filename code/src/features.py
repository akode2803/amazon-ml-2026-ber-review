"""
Stage 4: field-aware pair features.

Two complementary halves:

*Set features* -- computed in numba by merging two sorted hash arrays per
field.  Every field contributes not just agreement but **disagreement**:
`max_idf` of the tokens present on only one side is a contradiction signal, and
it is what separates "Rapid Telecom Systems / 14593 Lake Meadows Dr" from
"Rapid Telecom Systems / 92 Oak St".  Counts are also exported raw so that
missingness is visible to the model rather than silently looking like
agreement (two empty addresses have Jaccard 0/0, which we encode as a flag, not
as similarity 1).

*String features* -- rapidfuzz `cpdist`, batched over the whole pair list with
released GIL, so 35M pairs cost seconds rather than minutes.

All idf weights come from document frequencies pooled over S1+S2+S3 **within a
country**, so a token's weight means the same thing on both sides of a pair and
the feature distributions are comparable across countries -- which is what lets
a model trained on US+India transfer to France.
"""
from __future__ import annotations

import os
import sys

import numba
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import h  # noqa: E402

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"

FIELDS = ["name_toks", "name_skels", "addr_toks", "addr_skels", "nums", "geo"]
SHORT = ["nt", "ns", "at", "as", "nm", "geo"]


_HMEMO: dict = {}


def hm(t: str) -> int:
    """Memoised FNV hash.  A partition tokenises ~10M token occurrences over
    ~1M distinct strings, so memoising turns 10M python-level hash loops into
    10M dict lookups -- the difference between minutes and seconds."""
    v = _HMEMO.get(t)
    if v is None:
        v = _HMEMO[t] = h(t)
    return v


def _csr_from_col(vals):
    ptr = np.zeros(len(vals) + 1, np.int64)
    out = []
    for i, v in enumerate(vals):
        if v:
            out.extend({hm(x) for x in v.split("\t") if x})
        ptr[i + 1] = len(out)
    a = np.asarray(out, np.uint32)
    _sort_blocks(ptr, a)
    return ptr, a


@numba.njit(cache=True)
def _sort_blocks(ptr, a):
    for i in range(ptr.shape[0] - 1):
        lo, hi = ptr[i], ptr[i + 1]
        if hi - lo > 1:
            a[lo:hi] = np.sort(a[lo:hi])


class FieldStore:
    """Per-(split, country, source) field CSRs with pooled-idf weights."""

    def __init__(self, split, country, source, idf_tables):
        m = pl.read_parquet(f"{CACHE}/{split}/s{source}/meta.parquet")
        sel = (m["country"] == country).to_numpy().nonzero()[0]
        m = m[sel]
        self.rows = sel.astype(np.int32)
        self.n = len(sel)
        self.name_norm = m["name_norm"].to_list()
        self.addr_norm = m["addr_norm"].to_list()
        self.skel_norm = [s.replace("\t", "") for s in m["name_skels"].to_list()]
        self.tail = m["tail"].to_list()
        self.ptr, self.val, self.idf = {}, {}, {}
        for f, sh in zip(FIELDS, SHORT):
            p, v = _csr_from_col(m[f].to_list())
            self.ptr[sh], self.val[sh] = p, v
            vocab, w = idf_tables[sh]
            j = np.searchsorted(vocab, v)
            j = np.clip(j, 0, max(len(vocab) - 1, 0))
            self.idf[sh] = np.where(vocab[j] == v, w[j], w.max() if len(w) else 1.0
                                    ).astype(np.float32)
        self.has_addr = np.array([len(a) > 0 for a in self.addr_norm], np.int8)
        # largest building number as an integer (-1: none) -- lets stage 2
        # tell a typo (522->22, 7327->7317) from a neighbouring building (522->535)
        self.num_max = (m["nums"].fill_null("").str.split("\t")
                        .list.eval(pl.element().cast(pl.Int64, strict=False))
                        .list.max().fill_null(-1).to_numpy().astype(np.int64))
        z = np.load(f"{CACHE}/{split}/sig_{country}_s{source}.npz")
        self.sig = {k: z[k] for k in z.files}
        assert len(self.sig["ad"]) == self.n, "signature table out of sync"
        del m


def build_idf_tables(split, country):
    """Document frequency pooled over S1+S2+S3 of one country -> idf per field.

    Pooling across sources is what makes a weight mean the same thing on both
    sides of a pair; doing it per country keeps the scale comparable across
    countries, which is what lets a US+India model transfer to France.
    """
    tables = {}
    for f, sh in zip(FIELDS, SHORT):
        counts, n = None, 0
        for s in (1, 2, 3):
            m = (pl.scan_parquet(f"{CACHE}/{split}/s{s}/meta.parquet")
                 .filter(pl.col("country") == country)
                 .select(pl.col(f).str.split("\t").alias("t"))
                 .with_row_index("r").explode("t")
                 .filter(pl.col("t") != "")
                 .unique(subset=["r", "t"])          # df counts records, not tokens
                 .group_by("t").len().collect())
            n += (pl.scan_parquet(f"{CACHE}/{split}/s{s}/meta.parquet")
                  .filter(pl.col("country") == country)
                  .select(pl.len()).collect().item())
            counts = m if counts is None else pl.concat([counts, m])
            del m
        counts = counts.group_by("t").agg(pl.col("len").sum())
        toks = counts["t"].to_list()
        vocab = np.fromiter((hm(t) for t in toks), np.uint32, len(toks))
        df = counts["len"].to_numpy()
        o = np.argsort(vocab, kind="stable")
        vocab, df = vocab[o], df[o]
        # hash collisions merge two tokens' postings: sum their df so the idf
        # stays a valid (conservative) weight rather than a random one
        first = np.ones(len(vocab), bool)
        if len(vocab) > 1:
            first[1:] = vocab[1:] != vocab[:-1]
        gid = np.cumsum(first) - 1
        dfm = np.zeros(gid[-1] + 1 if len(gid) else 0, np.int64)
        np.add.at(dfm, gid, df)
        vocab = vocab[first]
        idf = np.log(1.0 + n / np.maximum(dfm, 1)).astype(np.float32)
        tables[sh] = (vocab, idf)
        print(f"      idf[{sh}] {len(vocab):,} terms over {n:,} records", flush=True)
    return tables


N_SET = 9  # features emitted per field


@numba.njit(parallel=True, cache=True)
def _set_feats(qi, ci, qptr, qval, qidf, cptr, cval, cidf, out, base):
    for t in numba.prange(qi.shape[0]):
        a0, a1 = qptr[qi[t]], qptr[qi[t] + 1]
        b0, b1 = cptr[ci[t]], cptr[ci[t] + 1]
        na, nb = a1 - a0, b1 - b0
        inter = 0
        wi = 0.0
        mx_sh = 0.0
        mx_a = 0.0
        mx_b = 0.0
        wa = 0.0
        wb = 0.0
        i, j = a0, b0
        while i < a1 and j < b1:
            if qval[i] == cval[j]:
                w = qidf[i]
                inter += 1
                wi += w
                if w > mx_sh:
                    mx_sh = w
                wa += w
                wb += cidf[j]
                i += 1
                j += 1
            elif qval[i] < cval[j]:
                w = qidf[i]
                wa += w
                if w > mx_a:
                    mx_a = w
                i += 1
            else:
                w = cidf[j]
                wb += w
                if w > mx_b:
                    mx_b = w
                j += 1
        while i < a1:
            w = qidf[i]
            wa += w
            if w > mx_a:
                mx_a = w
            i += 1
        while j < b1:
            w = cidf[j]
            wb += w
            if w > mx_b:
                mx_b = w
            j += 1
        union = na + nb - inter
        wun = wa + wb - wi
        out[t, base + 0] = inter
        out[t, base + 1] = inter / union if union > 0 else -1.0
        out[t, base + 2] = inter / min(na, nb) if min(na, nb) > 0 else -1.0
        out[t, base + 3] = wi
        out[t, base + 4] = wi / wun if wun > 0 else -1.0
        out[t, base + 5] = mx_sh
        out[t, base + 6] = mx_a          # rare token only the S1 record has
        out[t, base + 7] = mx_b          # rare token only the candidate has
        out[t, base + 8] = min(na, nb)   # missingness / field size


def set_feature_names():
    names = []
    for sh in SHORT:
        names += [f"{sh}_inter", f"{sh}_jac", f"{sh}_cont", f"{sh}_widf",
                  f"{sh}_wjac", f"{sh}_mxsh", f"{sh}_mxq", f"{sh}_mxc",
                  f"{sh}_minn"]
    return names


STR_SCORERS = [("nm_tsr", fuzz.token_set_ratio, "name"),
               ("nm_tso", fuzz.token_sort_ratio, "name"),
               ("nm_rat", fuzz.ratio, "name"),
               ("nm_par", fuzz.partial_ratio, "name"),
               ("nm_jw", JaroWinkler.normalized_similarity, "name"),
               ("sk_rat", fuzz.ratio, "skel"),
               ("sk_tso", fuzz.token_sort_ratio, "skel"),
               ("ad_tsr", fuzz.token_set_ratio, "addr"),
               ("ad_rat", fuzz.ratio, "addr"),
               ("ad_par", fuzz.partial_ratio, "addr"),
               ("tl_rat", fuzz.ratio, "tail")]

CTX_NAMES = ["blk_score", "blk_rank", "blk_rel", "blk_margin", "n_cand",
             "tgt_best", "tgt_rel", "tgt_rank", "src_is3",
             "q_hasaddr", "c_hasaddr", "both_addr", "len_ratio_nm",
             "len_ratio_ad", "fwd_rank", "rev_rank"]


# record-level information content (sigs.py): is the *whole* address / name
# identical, and how many records / distinct S1 entities share it
SIG_NAMES = ["ad_eq", "adl_eq", "nm_eq",
             "q_ad_all", "q_ad_s1", "c_ad_all", "c_ad_s1",
             "q_adl_s1", "c_adl_s1", "q_nm_all", "q_nm_s1", "c_nm_s1"]


USE_SIG = os.environ.get("BER_SIG", "1") == "1"


def feature_names():
    return (set_feature_names() + [s[0] for s in STR_SCORERS] + CTX_NAMES
            + (SIG_NAMES if USE_SIG else []))


def compute(qstore, cstore, qi, ci, ctx):
    """Feature matrix for pairs (qstore[qi[t]], cstore[ci[t]])."""
    nf = len(feature_names())
    out = np.zeros((len(qi), nf), np.float32)
    for b, sh in enumerate(SHORT):
        _set_feats(qi, ci, qstore.ptr[sh], qstore.val[sh], qstore.idf[sh],
                   cstore.ptr[sh], cstore.val[sh], cstore.idf[sh],
                   out, b * N_SET)
    col = N_SET * len(SHORT)
    src = {"name": (qstore.name_norm, cstore.name_norm),
           "addr": (qstore.addr_norm, cstore.addr_norm),
           "skel": (qstore.skel_norm, cstore.skel_norm),
           "tail": (qstore.tail, cstore.tail)}
    cache = {}
    for k, (qa, ca) in src.items():
        cache[k] = ([qa[i] for i in qi], [ca[i] for i in ci])
    for name, scorer, field in STR_SCORERS:
        a, b = cache[field]
        out[:, col] = process.cpdist(a, b, scorer=scorer, workers=-1,
                                     dtype=np.float32)
        col += 1
    for j in range(len(CTX_NAMES)):
        out[:, col + j] = ctx[:, j]
    col += len(CTX_NAMES)
    if not USE_SIG:
        return out
    qs, cs = qstore.sig, cstore.sig
    for j, k in enumerate(("ad", "adl", "nm")):
        a, b = qs[k][qi], cs[k][ci]
        out[:, col + j] = ((a == b) & (a != 0)).astype(np.float32)
    col += 3
    lg = np.log1p
    for j, arr in enumerate((lg(qs["ad_all"][qi]), qs["ad_s1"][qi],
                             lg(cs["ad_all"][ci]), cs["ad_s1"][ci],
                             qs["adl_s1"][qi], cs["adl_s1"][ci],
                             lg(qs["nm_all"][qi]), qs["nm_s1"][qi],
                             cs["nm_s1"][ci])):
        out[:, col + j] = arr
    return out
