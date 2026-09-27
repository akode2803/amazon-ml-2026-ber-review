"""
The competition metric, and the decision rule that maximises it.

Per S1 entity, with  c = #correct predicted, n = #predicted, T = #true:

    P = c/n,  R = c/T
    F0.5 = 1.25 P R / (0.25 P + R)
         = 1.25 (c/n)(c/T) / (0.25 c/n + c/T)
         = 1.25 c / (0.25 T + n)                       <-- exact, for c > 0

with the convention F=1 when n=T=0 and F=0 when exactly one of n,T is 0.

Two consequences drive the whole decision layer:

1. The denominator does not depend on *which* candidates we pick, only on how
   many.  So the expected score of a predicted set S is linear in the
   per-candidate match probabilities:

       E[F0.5(S)] = 1.25 * sum_{i in S} p_i / (|S| + 0.25 T)

   The optimal S is therefore a prefix of the candidates sorted by p, and the
   greedy stopping rule is exact: append candidate j iff

       p_j  >  (sum of already-accepted p) / (|S| + 0.25 T)

   i.e. iff its probability beats the current objective value / 1.25.
   That replaces an arbitrary global threshold ("predict if p > 0.7") with a
   per-entity rule that adapts to how many good candidates the entity has.

2. Predicting nothing scores P(T=0), not 0.  Abstention is worth real points:
   5.6% of train entities are singletons.
"""
from __future__ import annotations

import numpy as np


def f05_entity(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    c = len(pred & true)
    if c == 0:
        return 0.0
    return 1.25 * c / (0.25 * len(true) + len(pred))


def macro_f05(preds: dict, truths: dict) -> float:
    """Mean of the per-entity F0.5 over every S1 entity in `truths`."""
    return float(np.mean([f05_entity(preds.get(k, set()), v)
                          for k, v in truths.items()]))


def greedy_select(probs, t_hat, p_singleton=None, min_p=0.0):
    """Expected-F0.5-optimal prefix of `probs` (descending).

    Returns the number of candidates to accept.  `t_hat` estimates the entity's
    true match count T; `p_singleton` is P(T=0) -- the score the empty
    prediction earns -- and is compared against the best non-empty expectation.
    """
    p = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    k = 0.25 * max(t_hat, 1e-6)
    run = 0.0
    best_v, best_n = 0.0, 0
    for j, pj in enumerate(p, start=1):
        if pj < min_p:
            break
        v = 1.25 * (run + pj) / (j + k)
        if v <= best_v:
            break                      # objective is unimodal in the prefix
        run += pj
        best_v, best_n = v, j
    if p_singleton is not None and p_singleton > best_v:
        return 0, best_v
    return best_n, best_v


def _selftest():
    rng = np.random.default_rng(0)
    # 1. closed form matches the definition
    for _ in range(2000):
        T = rng.integers(0, 8)
        true = set(rng.choice(200, T, replace=False).tolist())
        n = rng.integers(0, 8)
        pred = set(rng.choice(200, n, replace=False).tolist())
        c = len(pred & true)
        if pred and true:
            P, R = c / len(pred), c / len(true)
            ref = 0.0 if (P + R) == 0 else 1.25 * P * R / (0.25 * P + R)
        elif not pred and not true:
            ref = 1.0
        else:
            ref = 0.0
        assert abs(ref - f05_entity(pred, true)) < 1e-12, (pred, true)
    # 2. the README's worked example
    assert abs(f05_entity({"a", "b", "c"}, {"a", "c"}) - 0.714285714) < 1e-6

    # 3. greedy prefix == exhaustive best prefix, in expectation
    for _ in range(4000):
        m = rng.integers(1, 9)
        p = np.sort(rng.random(m))[::-1]
        T = rng.uniform(0.5, 6)
        n, v = greedy_select(p, T)
        vals = [1.25 * p[:j].sum() / (j + 0.25 * T) for j in range(1, m + 1)]
        jb = int(np.argmax(vals)) + 1
        assert abs(vals[jb - 1] - v) < 1e-9 and n == jb, (p, T, n, jb)
    print("metric self-test OK: closed form, worked example, greedy optimality")

    # 4. what the rule actually does
    print("\n  marginal value of one more prediction (T=3.46, the train mean):")
    print("   accepted so far |  p needed to be worth adding")
    run, T = 0.0, 3.46
    for j in range(1, 7):
        thr = run / (j - 1 + 0.25 * T) if j > 1 else 0.0
        print(f"        {j-1:>5d}       |   p > {thr:.3f}")
        run += 0.95
    print("\n  a fixed global threshold cannot express this: the bar rises as")
    print("  an entity accumulates confident matches.")


if __name__ == "__main__":
    _selftest()
