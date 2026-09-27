"""
Stage 3 (GPU): a multilingual cross-encoder over the (S1, candidate) text of the
pairs stage 2 is unsure about.

Why.  Stage 1/2 see hand-built similarity features learned on US and India;
they transfer poorly to a country with no labels (France).  A pretrained
multilingual encoder (intfloat/multilingual-e5-small, cached locally) already
knows French/Hindi vocabulary and transliteration, so a pair scorer fine-tuned
on it should carry over better.  It only re-scores the uncertain band
BAND_LO <= p2 <= BAND_HI; confident pairs keep their stage-2 probability.

Steps (each writes under work/xenc):
    python xenc.py data     training sample + val/test band pairs
    python xenc.py train    fine-tune on GPU
    python xenc.py score    cross-encoder logits for val/test band pairs
    python xenc.py blend    logistic blend fitted on val; writes blended dumps

Env: VAL_TAG (stack TAG whose val npz is blended, s2xP2), DUMPS (test dumps
to blend, "fr_p2"), N_TRAIN, EPOCHS, BATCH, LR.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402

ROOT = M.ROOT
CACHE = f"{ROOT}/work/cache"
OUT = f"{ROOT}/work/xenc"
DUMPD = f"{ROOT}/work/dumps"
MODEL_NAME = "intfloat/multilingual-e5-small"
BAND_LO = float(os.environ.get("BAND_LO", "0.02"))   # <0.02 lets the CE rescue dropped pairs
BAND_HI = float(os.environ.get("BAND_HI", "0.98"))   # >1 re-scores every kept pair
BT = os.environ.get("BAND_TAG", "")                   # file suffix for a non-default band
VAL_TAG = os.environ.get("VAL_TAG", "s2xP2")
DUMPS = [d for d in os.environ.get("DUMPS", "fr_p2").split(",") if d]
N_TRAIN = int(os.environ.get("N_TRAIN", "2000000"))
MAXLEN = 96
# several cross-encoders side by side: MODEL_TAG names this one's files,
# CE_TAGS lists the ones whose logits the blend averages ("" = the first)
MODEL_TAG = os.environ.get("MODEL_TAG", "")
CE_TAGS = os.environ.get("CE_TAGS", MODEL_TAG).split(",")
os.makedirs(OUT, exist_ok=True)


def texts(split, src):
    d = pl.read_parquet(f"{CACHE}/{split}/s{src}/meta.parquet",
                        columns=["name_norm", "addr_norm"])
    return (d["name_norm"].fill_null("") + " ; "
            + d["addr_norm"].fill_null("")).to_numpy()


def rows_of(split, src, tid):
    """Global meta row of each target id."""
    ids = pl.read_parquet(f"{CACHE}/{split}/s{src}/meta.parquet",
                          columns=["id"])["id"].to_numpy().astype(np.int64)
    o = np.argsort(ids)
    sid = ids[o]
    j = np.minimum(np.searchsorted(sid, tid.astype(np.int64)), len(sid) - 1)
    assert np.all(sid[j] == tid), "unknown target id"
    return o[j].astype(np.int64)


def pair_texts(split, r1, rT, src):
    T1 = texts(split, 1)
    TT = {s: texts(split, s) for s in (2, 3)}
    a = T1[r1]
    b = np.where(src == 2, TT[2][np.minimum(rT, len(TT[2]) - 1)],
                 TT[3][np.minimum(rT, len(TT[3]) - 1)])
    return a, b


# --------------------------------------------------------------------- data
def cmd_data():
    t0 = time.time()
    s1row, pq, ps, tgt_row = M.gt_arrays("train")
    ids1 = pl.read_parquet(f"{CACHE}/train/s1/meta.parquet",
                           columns=["id"])["id"].to_numpy()
    fold = M.fold_of(ids1)
    rng = np.random.default_rng(int(os.environ.get("SEED", "7")))   # new seed -> new sample
    parts = [(c, s) for c in ("US", "India") for s in (2, 3)]
    R1, RT, SRC, Y = [], [], [], []
    for country, s in parts:
        z = np.load(f"{CACHE}/train/cand/c2_{country}_s{s}.npz")
        r1 = z["rows1"][z["q"]].astype(np.int64)
        rT = z["rowsT"][z["c"]].astype(np.int64)
        m = fold[r1] != 0                       # val fold never trains
        gsel = ps == s
        gk = np.unique((s1row[pq[gsel]].astype(np.int64) << 32)
                       | tgt_row[gsel].astype(np.int64))
        y = np.isin((r1 << 32) | rT, gk)
        pc = z["pc"]
        # oversample what the cheap ranker finds hard; keep some easy pairs
        w = np.where((pc > BAND_LO) & (pc < BAND_HI), 1.0, 0.15) * m
        k = min(N_TRAIN // len(parts), int(m.sum()))
        idx = rng.choice(len(w), k, replace=False, p=w / w.sum())
        R1.append(r1[idx]); RT.append(rT[idx]); Y.append(y[idx])
        SRC.append(np.full(k, s, np.int8))
        print(f"  train {country} S{s}: {k:,} pairs, {y[idx].mean() * 100:.1f}% pos",
              flush=True)
    # self-training for the text model: an unlabelled country's test pairs,
    # pseudo-labelled from a stage-2 dump (same rule as stack.py PSEUDO with
    # "explained" negatives).  PSEUDO_DUMP="France:fr_p2:600000"
    for spec in [s for s in os.environ.get("PSEUDO_DUMP", "").split(";") if s]:
        country, name, npl = spec.split(":")
        z = np.load(f"{DUMPD}/{name}.npz")
        dsrc, drow = z["src"], z["row"].astype(np.int64)
        dp = z["p"].astype(np.float64)
        drT = np.empty(len(dp), np.int64)
        for s in (2, 3):
            m = dsrc == s
            drT[m] = rows_of("test", s, z["tid"][m])
        tkey = (dsrc.astype(np.int64) << 40) | drT
        o = np.lexsort((-dp, tkey))
        win = np.zeros(len(dp), bool)
        win[o[np.concatenate([[True], tkey[o][1:] != tkey[o][:-1]])]] = True
        conf = win & (dp >= 0.98)
        pos_rows, claimed = np.unique(drow[conf]), np.unique(tkey[conf])
        dk = np.sort((dsrc.astype(np.int64) << 60) | (drow << 30) | drT)
        for s in (2, 3):
            c = np.load(f"{CACHE}/test/cand/c2_{country}_s{s}.npz")
            r1 = c["rows1"][c["q"]].astype(np.int64)
            rT = c["rowsT"][c["c"]].astype(np.int64)
            k = (np.int64(s) << 60) | (r1 << 30) | rT
            in_dump = np.isin(k, dk)            # dump holds every pair with p2>=0.02
            neg = ~in_dump & (np.isin(r1, pos_rows)
                              | np.isin((np.int64(s) << 40) | rT, claimed))
            pm = conf & (dsrc == s)
            pr1, prT = drow[pm], drT[pm]
            nidx = neg.nonzero()[0]
            n_each = int(npl) // 4
            pi = rng.choice(len(pr1), min(n_each, len(pr1)), replace=False)
            ni = rng.choice(nidx, min(n_each, len(nidx)), replace=False)
            R1 += [pr1[pi], r1[ni]]; RT += [prT[pi], rT[ni]]
            Y += [np.ones(len(pi), bool), np.zeros(len(ni), bool)]
            SRC += [np.full(len(pi) + len(ni), s, np.int8)]
            print(f"  pseudo {country} S{s}: {len(pi):,} pos + {len(ni):,} neg", flush=True)
    tsplit = np.concatenate([np.zeros(sum(len(x) for x in R1[:len(parts)]), np.int8),
                             np.ones(sum(len(x) for x in R1[len(parts):]), np.int8)])
    np.savez(f"{OUT}/train_pairs{MODEL_TAG}.npz", r1=np.concatenate(R1),
             rT=np.concatenate(RT), src=np.concatenate(SRC), y=np.concatenate(Y),
             split=tsplit)

    # val band: the stage-2 val pairs the blend is fitted on
    v = np.load(f"{ROOT}/work/models/val_scores_{VAL_TAG}.npz")
    idx = ((v["p2"] >= BAND_LO) & (v["p2"] <= BAND_HI)).nonzero()[0]
    src = v["src"][idx]
    rT = np.empty(len(idx), np.int64)
    for s in (2, 3):
        m = src == s
        rT[m] = rows_of("train", s, v["tid"][idx][m])
    np.savez(f"{OUT}/val_band{BT}_{VAL_TAG}.npz", idx=idx,
             r1=s1row[v["gt"][idx]].astype(np.int64), rT=rT, src=src)
    print(f"  val band ({VAL_TAG}): {len(idx):,} of {len(v['p2']):,} pairs", flush=True)

    for name in DUMPS:
        z = np.load(f"{DUMPD}/{name}.npz")
        idx = ((z["p"] >= BAND_LO) & (z["p"] <= BAND_HI)).nonzero()[0]
        src = z["src"][idx]
        rT = np.empty(len(idx), np.int64)
        for s in (2, 3):
            m = src == s
            rT[m] = rows_of("test", s, z["tid"][idx][m])
        np.savez(f"{OUT}/test_band{BT}_{name}.npz", idx=idx,
                 r1=z["row"][idx].astype(np.int64), rT=rT, src=src)
        print(f"  test band ({name}): {len(idx):,} of {len(z['p']):,} pairs", flush=True)
    print(f"data done {time.time() - t0:.0f}s")


# -------------------------------------------------------------------- model
def _enc(tok, a, b):
    return tok(list(a), list(b), truncation=True, max_length=MAXLEN,
               padding=True, return_tensors="pt")


def _predict(net, tok, a, b, dev, bs=512):
    import torch
    lens = np.fromiter((len(x) + len(y) for x, y in zip(a, b)), np.int64, len(a))
    o = np.argsort(lens)            # length-sorted batches pad far less
    out = np.empty(len(a), np.float32)
    with torch.no_grad():
        for i in range(0, len(o), bs):
            bi = o[i:i + bs]
            x = _enc(tok, a[bi], b[bi]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[bi] = net(**x).logits.squeeze(-1).float().cpu().numpy()
    return out


def cmd_train():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.time()
    epochs = float(os.environ.get("EPOCHS", "1"))
    bs = int(os.environ.get("BATCH", "128"))
    lr = float(os.environ.get("LR", "4e-5"))
    z = np.load(f"{OUT}/train_pairs{MODEL_TAG}.npz")
    y = z["y"].astype(np.float32)
    sp = z["split"] if "split" in z.files else np.zeros(len(y), np.int8)
    a = np.empty(len(y), object); b = np.empty(len(y), object)
    for k, split in ((0, "train"), (1, "test")):    # pseudo pairs are test records
        m = sp == k
        if m.any():
            a[m], b[m] = pair_texts(split, z["r1"][m], z["rT"][m], z["src"][m])
    rng = np.random.default_rng(int(os.environ.get("SEED", "7")) + 4)
    perm = rng.permutation(len(y))
    hold, tr = perm[:20000], perm[20000:]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    net = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    dev = torch.device("cuda")
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = int(len(tr) * epochs) // bs
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * max(0.0, 1 - i / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train()
    order = np.concatenate([rng.permutation(tr) for _ in range(int(np.ceil(epochs)))])
    for i in range(steps):
        bi = order[i * bs:(i + 1) * bs]
        x = _enc(tok, a[bi], b[bi]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(**x).logits.squeeze(-1)
        loss = lossf(out.float(), torch.from_numpy(y[bi]).to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if i % 500 == 0 or i == steps - 1:
            print(f"  step {i:>6}/{steps}  loss {loss.item():.4f}  "
                  f"{(i + 1) * bs / (time.time() - t0):.0f} pairs/s", flush=True)
    net.eval()
    from sklearn.metrics import log_loss, roc_auc_score
    pr = 1 / (1 + np.exp(-_predict(net, tok, a[hold], b[hold], dev)))
    print(f"holdout: logloss {log_loss(y[hold], pr):.4f}  "
          f"auc {roc_auc_score(y[hold], pr):.5f}")
    net.save_pretrained(f"{OUT}/model{MODEL_TAG}")
    tok.save_pretrained(f"{OUT}/model{MODEL_TAG}")
    print(f"train done {time.time() - t0:.0f}s")


def cmd_score():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(f"{OUT}/model{MODEL_TAG}")
    net = AutoModelForSequenceClassification.from_pretrained(f"{OUT}/model{MODEL_TAG}")
    dev = torch.device("cuda")
    net.to(dev).eval()
    jobs = [("train", f"val_band{BT}_{VAL_TAG}")] + [("test", f"test_band{BT}_{d}") for d in DUMPS]
    for split, name in jobs:
        z = np.load(f"{OUT}/{name}.npz")
        a, b = pair_texts(split, z["r1"], z["rT"], z["src"])
        lg = _predict(net, tok, a, b, dev)
        np.save(f"{OUT}/ce{MODEL_TAG}_{name}.npy", lg)
        print(f"  scored {name}: {len(lg):,} pairs  {time.time() - t0:.0f}s", flush=True)


# -------------------------------------------------------------------- blend
def _logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cmd_blend():
    import waterfall as W
    from sklearn.linear_model import LogisticRegression
    v = np.load(f"{ROOT}/work/models/val_scores_{VAL_TAG}.npz")
    zb = np.load(f"{OUT}/val_band{BT}_{VAL_TAG}.npz")
    # CE_SEP=1: one blend input per cross-encoder (a much stronger model gets
    # its own weight); default: their mean logit as a single input
    sep = os.environ.get("CE_SEP", "0") == "1"

    def ce_cols(fmt):
        c = np.stack([np.load(fmt.format(t=t)) for t in CE_TAGS], 1)
        return c if sep else c.mean(1, keepdims=True)
    idx = zb["idx"]
    X = np.hstack([_logit(v["p2"][idx])[:, None],
                   ce_cols(f"{OUT}/ce{{t}}_val_band{BT}_{VAL_TAG}.npy")])
    lr = LogisticRegression(C=1.0, max_iter=500).fit(X, v["pos"][idx].astype(int))
    # CE_BOOST > 1 trusts the cross-encoder more than val says: on the LB it
    # transferred ~3x better than on val (v6ce).  OUT_TAG names the variant.
    boost = float(os.environ.get("CE_BOOST", "1"))
    ot = os.environ.get("OUT_TAG", BT)
    w = lr.coef_[0].copy(); w[1:] *= boost
    pred = lambda M: 1 / (1 + np.exp(-(M @ w + lr.intercept_[0])))  # noqa: E731
    print(f"blend coef logit(p2) {w[0]:.3f}  ce {np.round(w[1:], 3)} (boost {boost}, "
          f"tags {CE_TAGS}, sep {sep})  b {lr.intercept_[0]:.3f}")
    p3 = v["p2"].astype(np.float64).copy()
    p3[idx] = pred(X)
    np.savez(f"{ROOT}/work/models/val_scores_{VAL_TAG}ce{ot}.npz",
             **{k: v[k] for k in v.files if k != "p2"}, p2=p3.astype(np.float32))
    for tag in (VAL_TAG, VAL_TAG + "ce" + ot):
        for cn in os.environ.get("EVAL_CN", "US,India").split(","):
            os.environ["WF_COUNTRIES"] = cn
            V = W.load_val(tag, "p2")
            g, p, y, T = V["g"], V["p"], V["y"], V["T"]
            ex = W.exclusivity(p, V["src"], V["tid"])
            res = []
            for ts, ab in ((1.1, 1.0), (1.1, 4.0), (1.3, 1.0)):
                tk, _ = W.decide_pairs(g, p, len(T), ts, ab, ex)
                c = np.bincount(g[tk & y], minlength=len(T))
                n = np.bincount(g[tk], minlength=len(T))
                res.append(f"t{ts}/a{ab} {W.macro(c, n, T)[0]:.5f}")
            print(f"  {tag:<12} {cn:<6} " + "  ".join(res), flush=True)
    for name in DUMPS:
        z = np.load(f"{DUMPD}/{name}.npz")
        tb = np.load(f"{OUT}/test_band{BT}_{name}.npz")
        cet = ce_cols(f"{OUT}/ce{{t}}_test_band{BT}_{name}.npy")
        p = z["p"].astype(np.float64).copy()
        p[tb["idx"]] = pred(np.hstack([_logit(p[tb["idx"]])[:, None], cet]))
        np.savez(f"{DUMPD}/{name}ce{ot}.npz", **{k: z[k] for k in z.files if k != "p"},
                 p=p.astype(np.float32))
        print(f"  wrote {DUMPD}/{name}ce{ot}.npz  mean p on band "
              f"{z['p'][tb['idx']].mean():.3f} -> {p[tb['idx']].mean():.3f}")


if __name__ == "__main__":
    {"data": cmd_data, "train": cmd_train, "score": cmd_score,
     "blend": cmd_blend}[sys.argv[1]]()
