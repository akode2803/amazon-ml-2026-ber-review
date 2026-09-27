import polars as pl
D="/Users/atharvapawar/Desktop/Amazon ML/student_resource/dataset/test/"
def load(p,c): return pl.read_csv(D+p,separator="\t",quote_char=None,
    schema_overrides={"entity_id":pl.Utf8,"business_name":pl.Utf8,"business_address":pl.Utf8,"country":pl.Utf8}).filter(pl.col("country")==c)
s1=load("test_source1.tsv","France"); s2=load("test_source2.tsv","France"); s3=load("test_source3.tsv","France")
print("France counts  S1",len(s1),"S2",len(s2),"S3",len(s3))
print("\n--- S1 France samples ---")
for r in s1.head(14).iter_rows(): print(f"  {r[1]!r}\n     {r[2]!r}")
print("\n--- S2 France samples ---")
for r in s2.head(14).iter_rows(): print(f"  {r[1]!r}\n     {r[2]!r}")
print("\n--- S3 France samples ---")
for r in s3.head(14).iter_rows(): print(f"  {r[1]!r}\n     {r[2]!r}")
# find a probable group: pick an S1, search S2/S3 for shared rare token
import re,collections
def toks(s): return set(re.findall(r"[a-zà-ÿ0-9]{4,}",(s or "").lower()))
cnt=collections.Counter()
for r in s2.head(400000).iter_rows(): cnt.update(toks(r[1]))
print("\n--- probable France groups (rare shared name token) ---")
shown=0
s2h=s2.head(400000); s3h=s3.head(400000)
for r in s1.head(3000).iter_rows():
    t=[x for x in toks(r[1]) if 1<=cnt[x]<=6]
    if not t: continue
    key=t[0]
    m2=[q for q in s2h.iter_rows() if key in toks(q[1])][:4]
    m3=[q for q in s3h.iter_rows() if key in toks(q[1])][:4]
    if not m2 and not m3: continue
    print(f"\n### S1 {r[1]!r}\n     {r[2]!r}")
    for q in m2: print(f"  S2 {q[1]!r}\n     {q[2]!r}")
    for q in m3: print(f"  S3 {q[1]!r}\n     {q[2]!r}")
    shown+=1
    if shown>=5: break
