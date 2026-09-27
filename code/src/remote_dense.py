"""
India dense-retrieval rescue, step 2 (RunPod): embed raw record text with
intfloat/multilingual-e5-large (bi-encoder, mean pooling, "query: " prefix on
both sides -- a symmetric match task) and take each S1 record's top-K nearest
S2 and S3 records by cosine similarity, per split, from an approximate
(FAISS IVF) index -- no all-pairs comparison.

    python remote_dense.py      # dense_{split}_s{1,2,3}.parquet -> dense_knn_{split}_s{2,3}.npz
Env: DMODEL, K, WORKERS.
"""
from __future__ import annotations

import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

DMODEL = os.environ.get("DMODEL", "intfloat/multilingual-e5-large")
K = int(os.environ.get("K", "10"))
WORKERS = int(os.environ.get("WORKERS", "16"))


def embed(net, tok, texts, bs=1024):
    lens = np.fromiter((len(t) for t in texts), np.int64, len(texts))
    o = np.argsort(lens)
    batches = [o[i:i + bs].tolist() for i in range(0, len(o), bs)]

    def coll(idx):
        x = tok(["query: " + texts[i] for i in idx], truncation=True, max_length=64,
                padding=True, return_tensors="pt")
        x["idx"] = torch.tensor(idx)
        return x
    dl = DataLoader(batches, batch_size=None, num_workers=WORKERS, prefetch_factor=4,
                    collate_fn=coll)
    out = torch.empty((len(texts), net.config.hidden_size), dtype=torch.float16, device="cuda")
    with torch.no_grad():
        for x in dl:
            idx = x.pop("idx").cuda()
            x = {k: v.cuda(non_blocking=True) for k, v in x.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = net(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[idx] = torch.nn.functional.normalize(e.float(), dim=-1).half()
    return out


def knn(q, t, k):
    """Approximate nearest neighbours with a FAISS inverted-file index: each
    query scans NPROBE of NLIST clusters (~1% of the targets), so the cost grows
    with the candidate set, not with |S1| x |targets| -- the same scaling
    requirement as the lexical blocker."""
    import faiss
    tn = t.float().cpu().numpy(); qn = q.float().cpu().numpy()
    d = tn.shape[1]
    nlist = int(os.environ.get("NLIST", "4096"))
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(0)
    index.train(tn[rng.choice(len(tn), min(len(tn), 262144), replace=False)])
    index.add(tn)
    index.nprobe = int(os.environ.get("NPROBE", "48"))
    sim, j = index.search(qn, k)
    return sim, j


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(DMODEL)
    net = AutoModel.from_pretrained(DMODEL).to(torch.bfloat16).cuda().eval()
    for split in ("train", "test"):
        d1 = pl.read_parquet(f"dense_{split}_s1.parquet")
        q = embed(net, tok, d1["text"].to_list())
        print(f"{split} s1 embedded {len(d1):,}  {time.time() - t0:.0f}s", flush=True)
        for k in (2, 3):
            dk = pl.read_parquet(f"dense_{split}_s{k}.parquet")
            t = embed(net, tok, dk["text"].to_list())
            sim, j = knn(q, t, K)
            rows1 = d1["row"].to_numpy(); rowsT = dk["row"].to_numpy()
            ok = j.ravel() >= 0                      # IVF may return fewer than K
            np.savez(f"dense_knn_{split}_s{k}.npz", s1=np.repeat(rows1, K)[ok],
                     t=rowsT[j.ravel()[ok]], sim=sim.ravel()[ok].astype(np.float32),
                     rank=np.tile(np.arange(K, dtype=np.int8), len(rows1))[ok])
            print(f"{split} s{k}: {len(dk):,} embedded, knn done  {time.time() - t0:.0f}s", flush=True)
            del t
        del q
        torch.cuda.empty_cache()
    print("DENSE_DONE", flush=True)


if __name__ == "__main__":
    main()
