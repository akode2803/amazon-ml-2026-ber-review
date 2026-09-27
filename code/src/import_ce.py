"""
Map the RunPod cross-encoder's logits (one per deduplicated score pair, see
export_pairs.py) back onto every all-pairs band file, as ce{TAG}_<band>.npy,
so `xenc.py blend` can use them exactly like a locally scored CE.

    python import_ce.py [logits.npy] [tag]      (defaults: work/runpod/logits_ce5.npy, 5)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_pairs as E  # noqa: E402
import xenc as X  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else f"{E.OUT}/logits_ce5.npy"
    tag = sys.argv[2] if len(sys.argv) > 2 else "5"
    keys = np.load(f"{E.OUT}/score_keys.npy")
    lg = np.load(path)
    assert len(keys) == len(lg), (len(keys), len(lg))
    assert np.all(keys[1:] > keys[:-1]), "score keys must be sorted and unique"
    for split, name in E.BANDS:
        t = np.load(f"{X.OUT}/{name}.npz")
        k = E.key(split, t["src"], t["r1"], t["rT"])
        j = np.searchsorted(keys, k)
        assert np.all(keys[np.minimum(j, len(keys) - 1)] == k), f"missing pairs for {name}"
        np.save(f"{X.OUT}/ce{tag}_{name}.npy", lg[j])
        print(f"  ce{tag}_{name}: {len(k):,} pairs", flush=True)


if __name__ == "__main__":
    main()
