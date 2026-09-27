"""
India retrieval, fine-tuned (RunPod): train intfloat/multilingual-e5-large as a
matching embedder on ground-truth India matches (bienc_train.parquet; many are
English <-> native script), with in-batch negatives (symmetric InfoNCE: every
S1 text must pick its own target among the batch's targets and vice versa).
Then embed + approximate kNN (FAISS IVF, remote_dense.knn) exactly as the
off-the-shelf run, into dense2_knn_{split}_s{k}.npz.

    python remote_bienc.py
Env: N_PAIRS, BATCH, LR, TAU, WORKERS, FAISS_THREADS.
"""
from __future__ import annotations

import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

import remote_dense as RD

BASE = os.environ.get("DMODEL", "intfloat/multilingual-e5-large")
N_PAIRS = int(os.environ.get("N_PAIRS", "700000"))
BATCH = int(os.environ.get("BATCH", "256"))
LR = float(os.environ.get("LR", "2e-5"))
TAU = float(os.environ.get("TAU", "0.05"))
WORKERS = int(os.environ.get("WORKERS", "8"))
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True


def pool(net, x):
    h = net(**x).last_hidden_state
    m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
    return torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1).clamp(min=1)).float(), dim=-1)


FILES = os.environ.get("TRAIN_FILES", "bienc_train.parquet").split(",")
MODEL_OUT = os.environ.get("BI_OUT", "bienc_model")
MODEL_IN = os.environ.get("BI_MODEL", "bienc_model")
# retrieval jobs "split:suffix,..." -> dense_{split}_s{k}{suffix}.parquet
JOBS = [x.split(":") for x in os.environ.get("RETRIEVE", "train:,test:").split(",")]


def train():
    t0 = time.time()
    d = pl.concat([pl.read_parquet(f) for f in FILES]).sample(fraction=1.0, shuffle=True, seed=3)
    d = d.head(N_PAIRS)
    a, b = d["a"].to_list(), d["b"].to_list()
    tok = AutoTokenizer.from_pretrained(BASE)
    net = AutoModel.from_pretrained(BASE).cuda()
    steps = len(a) // BATCH
    batches = [list(range(i * BATCH, (i + 1) * BATCH)) for i in range(steps)]

    def coll(idx):
        ta = tok(["query: " + a[i] for i in idx], truncation=True, max_length=64,
                 padding=True, return_tensors="pt")
        tb = tok(["query: " + b[i] for i in idx], truncation=True, max_length=64,
                 padding=True, return_tensors="pt")
        return ta, tb
    dl = DataLoader(batches, batch_size=None, num_workers=WORKERS, prefetch_factor=4,
                    collate_fn=coll, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * max(0.0, 1 - i / steps))
    lab = torch.arange(BATCH, device="cuda")
    net.train()
    for i, (ta, tb) in enumerate(dl):
        ta = {k: v.cuda(non_blocking=True) for k, v in ta.items()}
        tb = {k: v.cuda(non_blocking=True) for k, v in tb.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ea, eb = pool(net, ta), pool(net, tb)
        s = ea @ eb.T / TAU
        loss = (torch.nn.functional.cross_entropy(s, lab)
                + torch.nn.functional.cross_entropy(s.T, lab)) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if i % 200 == 0 or i == steps - 1:
            print(f"  step {i:>5}/{steps}  loss {loss.item():.4f}  {time.time() - t0:.0f}s",
                  flush=True)
    net.save_pretrained(MODEL_OUT); tok.save_pretrained(MODEL_OUT)
    print(f"bienc train done {time.time() - t0:.0f}s", flush=True)


def retrieve():
    import faiss
    faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "12")))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_IN)
    net = AutoModel.from_pretrained(MODEL_IN).to(torch.bfloat16).cuda().eval()
    RD.WORKERS = WORKERS
    for split, suf in JOBS:
        d1 = pl.read_parquet(f"dense_{split}_s1{suf}.parquet")
        q = RD.embed(net, tok, d1["text"].to_list())
        for k in (2, 3):
            dk = pl.read_parquet(f"dense_{split}_s{k}{suf}.parquet")
            t = RD.embed(net, tok, dk["text"].to_list())
            print(f"{split} s{k}: embedded {len(dk):,}  {time.time() - t0:.0f}s", flush=True)
            sim, j = RD.knn(q, t, RD.K)
            rows1 = d1["row"].to_numpy(); rowsT = dk["row"].to_numpy()
            ok = j.ravel() >= 0
            np.savez(f"dense2_knn_{split}_s{k}{suf}.npz", s1=np.repeat(rows1, RD.K)[ok],
                     t=rowsT[j.ravel()[ok]], sim=sim.ravel()[ok].astype(np.float32),
                     rank=np.tile(np.arange(RD.K, dtype=np.int8), len(rows1))[ok])
            print(f"{split} s{k}: knn done  {time.time() - t0:.0f}s", flush=True)
            del t
        del q
        torch.cuda.empty_cache()
    print("BIENC_DONE", flush=True)


if __name__ == "__main__":
    steps = os.environ.get("STEPS", "train,retrieve").split(",")
    if "train" in steps:
        train()
    if "retrieve" in steps:
        retrieve()
