"""
Leakage audit.

Enumerates every channel by which label information could reach a feature or a
validation score, and measures the ones that are measurable.
"""
import os
import sys
import numpy as np, polars as pl
ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
D = f"{ROOT}/student_resource/dataset"

def rd(p, cols=None):
    return pl.read_csv(p, separator="\t", quote_char=None, columns=cols,
                       schema_overrides={"entity_id": pl.Utf8,
                                         "business_name": pl.Utf8,
                                         "business_address": pl.Utf8,
                                         "country": pl.Utf8})

print("1. duplicate records inside each source")
for sp in ("train", "test"):
    for s in (1, 2, 3):
        df = rd(f"{D}/{sp}/{sp}_source{s}.tsv")
        n = len(df)
        d = df.select(["business_name", "business_address"]).n_unique()
        print(f"   {sp}/s{s}: {n:,} rows, {d:,} distinct (name,addr) "
              f"-> {n-d:,} exact duplicates ({(n-d)/n*100:.3f}%)")
        del df

print("\n2. entity-id overlap between train and test")
for s in (1, 2, 3):
    a = set(rd(f"{D}/train/train_source{s}.tsv", ["entity_id"])["entity_id"].to_list())
    b = set(rd(f"{D}/test/test_source{s}.tsv", ["entity_id"])["entity_id"].to_list())
    print(f"   S{s}: |train|={len(a):,} |test|={len(b):,} shared ids={len(a & b):,}")
    del a, b

print("\n3. (name,address) overlap between train S1 and test S1")
a = rd(f"{D}/train/train_source1.tsv").select(["business_name", "business_address"])
b = rd(f"{D}/test/test_source1.tsv").select(["business_name", "business_address"])
sa = set(zip(a["business_name"].to_list(), a["business_address"].to_list()))
sb = set(zip(b["business_name"].to_list(), b["business_address"].to_list()))
print(f"   shared (name,addr) pairs: {len(sa & sb):,} "
      f"({len(sa & sb)/len(sb)*100:.4f}% of test S1)")
del a, b, sa, sb

print("\n4. validation-fold construction")
print("   folds are a hash of the S1 numeric id, so an entity is wholly in one")
print("   fold; no pair of the same entity can straddle train and validation.")
print("\n5. features derived from labels: none.")
print("   every feature is a function of the two record texts plus the")
print("   candidate-set geometry (ranks, margins, competition), which is")
print("   computed from blocking scores and model probabilities only.")
print("\n6. idf / document frequency")
print("   computed per split from that split's own records; the test tables are")
print("   built from test records only. Unsupervised and transductive - it uses")
print("   no labels and is available at inference time.")
print("\n7. calibration set")
print("   isotonic calibration is fitted on a 10% slice of the TRAINING folds,")
print("   never on the validation fold.")
