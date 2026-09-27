"""
Stage 5-6: the decision layer.

Given calibrated per-pair probabilities, choose each S1 entity's predicted set
so as to maximise the *expected value of the competition metric*, not to cross
a threshold.  From metric.py, for an entity with T true matches,

    E[F0.5(S)] = 1.25 * sum_{i in S} p_i / (|S| + 0.25 T)

so the optimum is a prefix of the p-sorted candidates and the stopping rule is
exact: accept candidate j iff p_j > (running sum) / (j - 1 + 0.25 T).
The acceptance bar therefore *rises* as an entity accumulates confident
matches -- a second 0.55 candidate is worth taking when the first was 0.55, and
not worth taking when the first three were 0.97.

T is unknown, so it is estimated as sum(p) inflated by the blocking recall
(matches we never retrieved still count in the denominator of the real metric).

Abstention is scored explicitly: an empty prediction earns 1.0 when the entity
really has no match, so we compare the best non-empty expectation against
P(T=0) ~= prod(1 - p_i), damped by a fitted factor for the chance that a true
match exists outside the candidate set.
"""
from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def decide_all(ptr, prob, t_scale, abstain, min_p, out_take, out_val):
    """For each entity (CSR group over p-sorted probabilities) pick a prefix.

    `ptr` delimits groups; `prob` must be sorted descending inside each group.
    Writes the number of accepted candidates into out_take.
    """
    for g in range(ptr.shape[0] - 1):
        lo, hi = ptr[g], ptr[g + 1]
        n = hi - lo
        if n == 0:
            out_take[g] = 0
            out_val[g] = 1.0
            continue
        tot = 0.0
        none = 1.0
        for i in range(lo, hi):
            tot += prob[i]
            none *= (1.0 - prob[i])
        t_hat = tot * t_scale
        if t_hat < 1e-6:
            t_hat = 1e-6
        k = 0.25 * t_hat
        run = 0.0
        best_v = 0.0
        best_n = 0
        for j in range(n):
            pj = prob[lo + j]
            if pj < min_p:
                break
            v = 1.25 * (run + pj) / (j + 1 + k)
            if v <= best_v:
                break
            run += pj
            best_v = v
            best_n = j + 1
        p0 = none * abstain
        if p0 > best_v:
            out_take[g] = 0
            out_val[g] = p0
        else:
            out_take[g] = best_n
            out_val[g] = best_v


@numba.njit(cache=True)
def f05_from_take(ptr, is_pos, take, T):
    """Macro-F0.5 given, per entity, its p-sorted candidate labels and prefix."""
    tot = 0.0
    for g in range(ptr.shape[0] - 1):
        lo = ptr[g]
        n = take[g]
        t = T[g]
        if n == 0:
            tot += 1.0 if t == 0 else 0.0
            continue
        if t == 0:
            continue                     # predicted something, truth empty -> 0
        c = 0
        for j in range(n):
            if is_pos[lo + j]:
                c += 1
        if c > 0:
            tot += 1.25 * c / (0.25 * t + n)
    return tot / (ptr.shape[0] - 1)


@numba.njit(cache=True)
def f05_threshold(ptr, prob, is_pos, T, thr):
    """Baseline: accept every candidate with p >= thr."""
    tot = 0.0
    for g in range(ptr.shape[0] - 1):
        lo, hi = ptr[g], ptr[g + 1]
        n = 0
        c = 0
        for i in range(lo, hi):
            if prob[i] >= thr:
                n += 1
                if is_pos[i]:
                    c += 1
        t = T[g]
        if n == 0:
            tot += 1.0 if t == 0 else 0.0
        elif t == 0 or c == 0:
            continue
        else:
            tot += 1.25 * c / (0.25 * t + n)
    return tot / (ptr.shape[0] - 1)


@numba.njit(cache=True)
def f05_top1(ptr, prob, is_pos, T, thr):
    """Baseline: accept only the single best candidate if it clears thr."""
    tot = 0.0
    for g in range(ptr.shape[0] - 1):
        lo, hi = ptr[g], ptr[g + 1]
        t = T[g]
        if hi == lo or prob[lo] < thr:
            tot += 1.0 if t == 0 else 0.0
            continue
        if t == 0:
            continue
        if is_pos[lo]:
            tot += 1.25 / (0.25 * t + 1)
    return tot / (ptr.shape[0] - 1)
