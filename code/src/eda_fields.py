import polars as pl, numpy as np, unicodedata, collections, random
D="/Users/atharvapawar/Desktop/Amazon ML/student_resource/dataset/"
def load(p): return pl.read_csv(D+p,separator="\t",quote_char=None,has_header=True,
        schema_overrides={"entity_id":pl.Utf8,"business_name":pl.Utf8,"business_address":pl.Utf8,"country":pl.Utf8})
for split in ["train","test"]:
    for s in [1,2,3]:
        df=load(f"{split}/{split}_source{s}.tsv")
        n=len(df)
        print(f"\n===== {split} S{s}  n={n:,} =====")
        print(df["country"].value_counts(sort=True).to_dicts())
        nm_null=df["business_name"].is_null().sum(); nm_empty=(df["business_name"].fill_null("").str.strip_chars()=="").sum()
        ad_null=df["business_address"].is_null().sum(); ad_empty=(df["business_address"].fill_null("").str.strip_chars()=="").sum()
        print(f"  name  missing/blank: {nm_empty:,} ({100*nm_empty/n:.2f}%)")
        print(f"  addr  missing/blank: {ad_empty:,} ({100*ad_empty/n:.2f}%)")
        print(f"  name len  mean {df['business_name'].fill_null('').str.len_chars().mean():.1f} p95 {df['business_name'].fill_null('').str.len_chars().quantile(0.95):.0f}")
        print(f"  addr len  mean {df['business_address'].fill_null('').str.len_chars().mean():.1f} p95 {df['business_address'].fill_null('').str.len_chars().quantile(0.95):.0f}")
        # script detection on a sample
        samp=df["business_name"].fill_null("").sample(min(200000,n),seed=0).to_list()
        sc=collections.Counter()
        for t in samp:
            has_dev=any('ऀ'<=c<='ॿ' for c in t)
            has_lat=any('a'<=c.lower()<='z' for c in t)
            sc['devanagari' if has_dev and not has_lat else 'mixed' if has_dev else 'latin' if has_lat else 'other']+=1
        print(f"  name script (200k sample): {dict(sc)}")
        # duplicate exact names
        print(f"  distinct raw names: {df['business_name'].n_unique():,}  ({100*df['business_name'].n_unique()/n:.1f}%)")
        print(f"  distinct raw (name,addr): {df.select(['business_name','business_address']).n_unique():,}")
