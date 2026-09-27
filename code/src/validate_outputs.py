"""
Submission self-check, run before the official validator.

Checks everything the scorer enforces plus the consistency rules the official
validator only warns about, and prints the blocking statistics the organisers
say they use to audit candidate_pairs.tsv (reduction ratio, candidates per S1).
"""
from __future__ import annotations

import os
import sys

DELIM = "\t"


def read_first_col(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def load(path, expect_col):
    errs = []
    rows = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(DELIM)
        if header != ["source1_entity_id", expect_col]:
            errs.append(f"{os.path.basename(path)}: bad header {header}")
            return rows, errs
        for ln, line in enumerate(f, 2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                errs.append(f"{os.path.basename(path)}: no tab at line {ln}")
                continue
            rest = rest.rstrip("\n")
            ids = rest.split(",") if rest else []
            if s1 in rows:
                errs.append(f"{os.path.basename(path)}: duplicate row {s1}")
            if len(ids) != len(set(ids)):
                errs.append(f"{os.path.basename(path)}: dup id inside list {s1}")
            for i in ids:
                if not i.startswith(("S2-", "S3-")):
                    errs.append(f"{os.path.basename(path)}: bad id {i} in {s1}")
                    break
            rows[s1] = set(ids)
    return rows, errs


def main(outdir, testdir):
    errs = []
    req = read_first_col(f"{testdir}/test_source1.tsv")
    print(f"required S1 entities: {len(req):,}")
    match, e = load(f"{outdir}/matching_results.tsv", "matched_entity_ids"); errs += e
    cand, e = load(f"{outdir}/candidate_pairs.tsv", "candidate_entity_ids"); errs += e

    for nm, d in (("matching_results", match), ("candidate_pairs", cand)):
        missing = req - set(d)
        extra = set(d) - req
        if missing:
            errs.append(f"{nm}: {len(missing):,} required S1 entities missing")
        if extra:
            errs.append(f"{nm}: {len(extra):,} rows with unknown S1 id")

    valid = read_first_col(f"{testdir}/test_source2.tsv") | \
        read_first_col(f"{testdir}/test_source3.tsv")
    print(f"valid S2/S3 ids: {len(valid):,}")
    for nm, d in (("matching_results", match), ("candidate_pairs", cand)):
        bad = set()
        for s1, ids in d.items():
            bad |= (ids - valid)
            if len(bad) > 5:
                break
        if bad:
            errs.append(f"{nm}: ids not present in the test set, e.g. "
                        f"{sorted(bad)[:5]}")

    notsub = [s1 for s1, ids in match.items() if ids - cand.get(s1, set())]
    if notsub:
        errs.append(f"{len(notsub):,} entities have matches that are not "
                    f"candidates, e.g. {notsub[:5]}")

    nm_ids = sum(len(v) for v in match.values())
    nc_ids = sum(len(v) for v in cand.values())
    n1 = len(req)
    n23 = len(valid)
    print(f"\ncandidate_pairs.tsv : {nc_ids:,} pairs, {nc_ids/n1:.2f}/S1, "
          f"{sum(1 for v in cand.values() if not v):,} empty rows")
    print(f"matching_results.tsv: {nm_ids:,} matches, {nm_ids/n1:.2f}/S1, "
          f"{sum(1 for v in match.values() if not v):,} abstentions "
          f"({sum(1 for v in match.values() if not v)/n1*100:.2f}%)")
    print(f"reduction ratio     : {n1*n23/max(nc_ids,1):,.0f}x  "
          f"(full cross product {n1*n23:,} -> {nc_ids:,})")

    if errs:
        print(f"\nFAIL - {len(errs)} issue(s):")
        for i, e in enumerate(errs, 1):
            print(f"  {i}. {e}")
        return 1
    print("\nPASS - both files satisfy every documented rule.")
    return 0


if __name__ == "__main__":
    o = sys.argv[1] if len(sys.argv) > 1 else "student_resource/output"
    t = sys.argv[2] if len(sys.argv) > 2 else "student_resource/dataset/test"
    sys.exit(main(o, t))
