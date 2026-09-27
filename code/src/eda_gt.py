"""EDA-1: ground-truth structure. Memory-lean: parse ids to int32 arrays."""
import numpy as np, collections, sys
D="/Users/atharvapawar/Desktop/Amazon ML/student_resource/dataset/train/"

s1_ids=[]; counts=[]; n2=[]; n3=[]
tgt_all=[]   # all matched numeric ids w/ source tag
with open(D+"train_ground_truth.tsv",encoding="utf-8") as f:
    next(f)
    for line in f:
        s1,_,rest=line.partition("\t")
        rest=rest.rstrip("\n")
        s1_ids.append(int(s1[3:]))
        if not rest:
            counts.append(0); n2.append(0); n3.append(0); continue
        ids=rest.split(",")
        c2=c3=0
        for m in ids:
            if m[1]=="2": c2+=1; tgt_all.append((2,int(m[3:])))
            elif m[1]=="3": c3+=1; tgt_all.append((3,int(m[3:])))
        counts.append(len(ids)); n2.append(c2); n3.append(c3)

counts=np.array(counts); n2=np.array(n2); n3=np.array(n3)
N=len(counts)
print(f"S1 entities in GT: {N:,}")
print(f"total matched ids: {counts.sum():,}   (S2 {n2.sum():,} / S3 {n3.sum():,})")
print(f"\n--- distribution of T = #true matches per S1 ---")
bc=np.bincount(counts)
cum=0
for t,c in enumerate(bc):
    if c==0: continue
    cum+=c
    print(f"  T={t:3d}  {c:>9,}  {100*c/N:6.2f}%   cum {100*cum/N:6.2f}%")
    if t>=15: print(f"  ... T>15: {(counts>15).sum():,}"); break
print(f"\nsingletons (T=0): {(counts==0).sum():,} = {100*(counts==0).mean():.3f}%")
print(f"mean T {counts.mean():.3f}  median {np.median(counts):.0f}  p95 {np.percentile(counts,95):.0f}  p99 {np.percentile(counts,99):.0f}  max {counts.max()}")
print(f"\n--- #S2 matches per S1 ---"); print(np.bincount(n2)[:12], " mean",round(n2.mean(),3))
print(f"--- #S3 matches per S1 ---"); print(np.bincount(n3)[:12], " mean",round(n3.mean(),3))
print(f"\nS1 with >=1 S2 and >=1 S3: {((n2>0)&(n3>0)).sum():,}")
print(f"S1 with only S2: {((n2>0)&(n3==0)).sum():,}   only S3: {((n2==0)&(n3>0)).sum():,}")

# one-to-one constraint check: does an S2/S3 id ever match >1 S1?
print("\n--- reuse of S2/S3 ids across multiple S1 entities ---")
cnt=collections.Counter(tgt_all)
dup=collections.Counter(v for v in cnt.values())
print("  matched-id multiplicity histogram:",dict(sorted(dup.items())[:8]))
print(f"  distinct matched ids: {len(cnt):,} of {len(tgt_all):,} references")
np.save("/Users/atharvapawar/Desktop/Amazon ML/work/cache/gt_counts.npy",counts)
np.save("/Users/atharvapawar/Desktop/Amazon ML/work/cache/gt_s1ids.npy",np.array(s1_ids,dtype=np.int64))
print("\nmax numeric id seen:",max(max(s1_ids), max(v for _,v in tgt_all)))
