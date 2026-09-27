"""
India dense-retrieval rescue, step 1 (local): export the raw record text (native
scripts kept -- the embedding model is what bridges "New Products" and
"न्यू प्रोडक्ट्स") for every India S1 / S2 / S3 record of the test split, and
of the train split's val-fold S1 entities + all train India targets (for an
honest measurement on the held-out fold).

Writes work/runpod/dense_{split}_s{k}.parquet with columns row (global cache
row) and text.  Cache rows follow the raw TSV row order.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402

ROOT = M.ROOT
DATA = f"{ROOT}/student_resource/dataset"
OUT = f"{ROOT}/work/runpod"
COUNTRY = os.environ.get("DENSE_COUNTRY", "India")
SPLITS = os.environ.get("SPLITS", "train,test").split(",")   # France: test only
SUF = os.environ.get("SUF", "")                              # e.g. _US, _FR


def main():
    os.makedirs(OUT, exist_ok=True)
    for split in SPLITS:
        for k in (1, 2, 3):
            raw = pl.read_csv(f"{DATA}/{split}/{split}_source{k}.tsv", separator="\t",
                              quote_char=None, infer_schema_length=0)
            ctry = pl.read_parquet(f"{M.CACHE}/{split}/s{k}/meta.parquet",
                                   columns=["id", "country"])
            assert len(raw) == len(ctry)
            assert (raw["entity_id"].str.slice(3).cast(pl.Int64).to_numpy()
                    == ctry["id"].to_numpy()).all(), "cache rows != raw rows"
            m = ctry["country"].to_numpy() == COUNTRY
            if split == "train" and k == 1:          # only held-out entities are queried
                m &= M.fold_of(ctry["id"].to_numpy()) == 0
            rows = np.flatnonzero(m)
            text = (raw["business_name"].fill_null("") + ", "
                    + raw["business_address"].fill_null("")).to_numpy()[rows]
            pl.DataFrame({"row": rows.astype(np.int64), "text": text.tolist()}) \
                .write_parquet(f"{OUT}/dense_{split}_s{k}{SUF}.parquet", compression="zstd")
            print(f"  {split} s{k}: {len(rows):,} {COUNTRY} records", flush=True)


if __name__ == "__main__":
    main()
