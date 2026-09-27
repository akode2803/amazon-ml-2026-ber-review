import polars as pl, random, sys
D="/Users/atharvapawar/Desktop/Amazon ML/student_resource/dataset/train/"
def load(p): return pl.read_csv(D+p,separator="\t",quote_char=None,
        schema_overrides={"entity_id":pl.Utf8,"business_name":pl.Utf8,"business_address":pl.Utf8,"country":pl.Utf8})
gt=pl.read_csv(D+"train_ground_truth.tsv",separator="\t",quote_char=None,schema_overrides={"source1_entity_id":pl.Utf8,"matched_entity_ids":pl.Utf8})
random.seed(7)
idx=random.sample(range(len(gt)),400)
sub=gt[idx]
need=set(sub["source1_entity_id"].to_list())
tgt=set()
for r in sub["matched_entity_ids"].to_list():
    if r: tgt.update(r.split(","))
s1=load("train_source1.tsv").filter(pl.col("entity_id").is_in(need))
s23=pl.concat([load("train_source2.tsv").filter(pl.col("entity_id").is_in(tgt)),
               load("train_source3.tsv").filter(pl.col("entity_id").is_in(tgt))])
m1={r[0]:r for r in s1.iter_rows()}; m23={r[0]:r for r in s23.iter_rows()}
ctry=sys.argv[1] if len(sys.argv)>1 else None
shown=0
for s1id,ml in sub.iter_rows():
    if s1id not in m1: continue
    r=m1[s1id]
    if ctry and r[3]!=ctry: continue
    print(f"\n### {r[3]} | {s1id}")
    print(f"  S1  NAME: {r[1]!r}\n      ADDR: {r[2]!r}")
    for mid in (ml.split(",") if ml else []):
        if mid in m23:
            q=m23[mid]; print(f"  {mid[:2]}  NAME: {q[1]!r}\n      ADDR: {q[2]!r}")
    shown+=1
    if shown>=18: break
