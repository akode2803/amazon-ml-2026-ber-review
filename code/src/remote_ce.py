"""
Large cross-encoder, run on a rented GPU (RunPod H100/A100).  Standalone: needs
only torch, transformers, polars, numpy, scikit-learn and the two parquet files
written by export_pairs.py.

    python remote_ce.py train   # train_ce5.parquet (a, b, y) -> model_large/
    python remote_ce.py score   # score_ce5.parquet (a, b)    -> logits_ce5.npy

Same recipe as xenc.py (pair classification, BCE, bf16, 1 epoch, linear decay),
with intfloat/multilingual-e5-large (560M, MIT) and parallel tokenisation so the
GPU is not starved.  Env: MODEL, BATCH, LR, EPOCHS, WORKERS.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.environ.get("MODEL", "intfloat/multilingual-e5-large")
MAXLEN = 96
BATCH = int(os.environ.get("BATCH", "128"))
LR = float(os.environ.get("LR", "2e-5"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
WORKERS = int(os.environ.get("WORKERS", "8"))
torch.backends.cuda.matmul.allow_tf32 = True


class Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return i


def collate(tok, ds):
    def f(idx):
        x = tok([ds.a[i] for i in idx], [ds.b[i] for i in idx], truncation=True,
                max_length=MAXLEN, padding=True, return_tensors="pt")
        x["idx"] = torch.tensor(idx)
        if ds.y is not None:
            x["y"] = torch.tensor(ds.y[idx], dtype=torch.float32)
        return x
    return f


def train():
    t0 = time.time()
    d = pl.read_parquet(os.environ.get("TRAIN_IN", "train_ce5.parquet"))
    a, b, y = d["a"].to_list(), d["b"].to_list(), d["y"].to_numpy().astype(np.float32)
    rng = np.random.default_rng(5)
    perm = rng.permutation(len(y))
    hold, tr = perm[:20000], perm[20000:]
    tok = AutoTokenizer.from_pretrained(MODEL)
    net = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    ds = Pairs(a, b, y)
    order = np.concatenate([rng.permutation(tr) for _ in range(int(np.ceil(EPOCHS)))])
    steps = int(len(tr) * EPOCHS) // BATCH
    batches = [order[i * BATCH:(i + 1) * BATCH].tolist() for i in range(steps)]
    dl = DataLoader(batches, batch_size=None, num_workers=WORKERS, prefetch_factor=4,
                    collate_fn=collate(tok, ds), persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * max(0.0, 1 - i / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train()
    for i, x in enumerate(dl):
        yb = x.pop("y").cuda(non_blocking=True); x.pop("idx")
        x = {k: v.cuda(non_blocking=True) for k, v in x.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(**x).logits.squeeze(-1)
        loss = lossf(out.float(), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if i % 500 == 0 or i == steps - 1:
            print(f"  step {i:>6}/{steps}  loss {loss.item():.4f}  "
                  f"{(i + 1) * BATCH / (time.time() - t0):.0f} pairs/s", flush=True)
    net.eval()
    from sklearn.metrics import log_loss, roc_auc_score
    lg = predict(net, tok, [a[i] for i in hold], [b[i] for i in hold])
    pr = 1 / (1 + np.exp(-lg))
    print(f"holdout: logloss {log_loss(y[hold], pr):.4f}  auc {roc_auc_score(y[hold], pr):.5f}")
    od = os.environ.get("OUT_MODEL", "model_large")
    net.save_pretrained(od); tok.save_pretrained(od)
    print(f"train done {time.time() - t0:.0f}s", flush=True)


def predict(net, tok, a, b, bs=1024):
    lens = np.fromiter((len(x) + len(y) for x, y in zip(a, b)), np.int64, len(a))
    o = np.argsort(lens)
    batches = [o[i:i + bs].tolist() for i in range(0, len(o), bs)]
    ds = Pairs(a, b)
    dl = DataLoader(batches, batch_size=None, num_workers=WORKERS, prefetch_factor=4,
                    collate_fn=collate(tok, ds))
    out = np.empty(len(a), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for j, x in enumerate(dl):
            idx = x.pop("idx").numpy()
            x = {k: v.cuda(non_blocking=True) for k, v in x.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[idx] = net(**x).logits.squeeze(-1).float().cpu().numpy()
            if j % 1000 == 0:
                print(f"  scored {min((j + 1) * bs, len(a)):,}/{len(a):,}  "
                      f"{(j + 1) * bs / (time.time() - t0):.0f} pairs/s", flush=True)
    return out


def score():
    d = pl.read_parquet(os.environ.get("SCORE_IN", "score_ce5.parquet"))
    md = os.environ.get("SCORE_MODEL", "model_large")
    tok = AutoTokenizer.from_pretrained(md)
    net = AutoModelForSequenceClassification.from_pretrained(md).cuda().eval()
    lg = predict(net, tok, d["a"].to_list(), d["b"].to_list())
    np.save(os.environ.get("SCORE_OUT", "logits_ce5.npy"), lg)
    print(f"score done: {len(lg):,} logits", flush=True)


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]]()
