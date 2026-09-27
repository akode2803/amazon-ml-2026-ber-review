import os
"""
Does normalising the IDF scale make the model transfer better to an unseen
country?

idf = log(1 + N/df) has a country-dependent ceiling, because N (records in that
country) differs: India 5.0M, US 7.5M, France 1.7M.  So every absolute
idf-weighted feature sits on a slightly different scale per country, and a
model trained on two countries can mis-read the third.  Dividing by log(1+N)
puts them all on [0,1].

We test it by rescaling the cached ablation matrices and re-running
leave-one-country-out, which is the only measurable proxy for France.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F, model as M, ablate as A

ROOT = os.environ.get("BER_ROOT", "C:/Users/LENOVO/Desktop/Amazon_ML/ber")
CACHE = f"{ROOT}/work/cache"
NAMES = F.feature_names()
IX = {n: i for i, n in enumerate(NAMES)}
# columns that are linear in idf and therefore carry the country scale
SCALED = [IX[n] for n in NAMES if n.endswith(("_widf", "_mxsh", "_mxq", "_mxc"))]
NREC = {"India": 5016534, "US": 7510506}     # from the idf build logs

tr = np.load(f"{CACHE}/abl_train.npz"); va = np.load(f"{CACHE}/abl_val.npz")
Xtr, ytr, ctr = tr["X"].copy(), tr["y"], tr["c"]
Xva, yva, gva, cva = va["X"].copy(), va["y"], va["g"], va["c"]
T = va["T"]
countries = M.countries_of("train")
print(f"rescaling {len(SCALED)} idf-linear columns: "
      f"{[NAMES[i] for i in SCALED][:6]} ...")
for k, cn in enumerate(countries):
    z = np.log1p(NREC[cn])
    Xtr[np.ix_((ctr == k).nonzero()[0], SCALED)] /= z
    Xva[np.ix_((cva == k).nonzero()[0], SCALED)] /= z

def fit_eval(X1, y1, X2, y2, g2, label):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score
    p = dict(objective="binary", learning_rate=0.08, num_leaves=63,
             min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
             bagging_freq=1, num_threads=10, verbosity=-1, max_bin=127)
    b = lgb.train(p, lgb.Dataset(X1, label=y1), num_boost_round=300)
    pr = b.predict(X2).astype(np.float32)
    f_exp, (f_thr, thr), *_ = A.run_decision(pr, g2, y2, T)
    print(f"  {label:<44} F0.5={f_exp:.5f}  best-thr={f_thr:.5f} "
          f"AP={average_precision_score(y2, pr):.4f}")
    return f_exp

print("\nleave-one-country-out, IDF scale NORMALISED:")
for k, cn in enumerate(countries):
    m = cva == k
    fit_eval(Xtr[ctr != k], ytr[ctr != k], Xva[m], yva[m], gva[m],
             f"train !={cn} -> val {cn}  (normalised)")
print("\n(compare with the un-normalised numbers in ablation.log)")
