# ASTRA HANDOFF — tear this pipeline apart

*Written by Opus, 27 Sep 2026 ~17:20 IST. Hard deadline 23:00 IST; code freeze ~21:00 IST.*

Astra — you are the reviewer we trust to find what we cannot see. We are **not** asking for a polite review or a
list of generic best practices. We want **every bug, leak, mis-wiring, silently wrong assumption and cheap win
that is costing us leaderboard points right now**, ranked by expected gain per minute of work. Assume the code
has bugs; our job is to find out where. If something looks fine, don't mention it. If you are unsure, say how to
check it in under 5 minutes. **Your time is limited (~2–3 h): put everything valuable in `astra/REVIEW_1.md`.**
Protocol: `COMMS.md`.

---

## 1. Task in five lines
- Source 1 (S1) = deduplicated reference businesses; S2/S3 = noisy records (0..many per S1, ~27% of train S2/S3
  and an estimated ~41% of test S2/S3 match nothing). Fields: `business_name`, `business_address`, `country`.
- Output: for every test S1, the list of matching S2/S3 ids (`matching_results.tsv`) + the candidate set the model
  scored (`candidate_pairs.tsv`, not scored on the LB but **reviewed for final ranking: smaller per entity is better,
  blocking must scale, no all-pairs**).
- Metric: **macro F0.5 per S1 entity** (singleton: 1 if we predict empty, else 0), averaged. Public LB = part of
  test; **final = private part**.
- Train: US + India (2.2M S1, 5.0M S2, 5.3M S3, 7.64M links). Test: US 663k, India 810k, **France 259k (zero-shot, no labels)**.
- Rules: no external data/APIs; models MIT/Apache ≤ 8B; code package must regenerate outputs.

## 2. Where we are (numbers you can hold us to)
| version | change | val (US / India) | **public LB** |
|---|---|---|---|
| v4 | lexical blocking + stage-1/2 LightGBM stack | 0.986 / 0.977 | 0.95949 |
| v6ce | + self-training on test + e5-small cross-encoder (CE) | 0.9887 / 0.9810 | 0.976768 |
| v10b | + CE on all kept pairs, CE1–CE3, CE weight × 1.5 | 0.9888 / 0.9812 | 0.980926 |
| v12s3 | + e5-large CE on rented H200 + **stage 3** LightGBM | 0.98952 / 0.98178 | 0.981866 |
| **v13_k3 (current best)** | + **India fine-tuned dense retrieval** (top-3 FAISS-IVF neighbours not in lexical set) | 0.98952 / **0.99066** | **0.985863** |
| target | top 25 | | **0.9909** |

- Validation = entity fold 0 of train (20%); US/India only. **Validation historically over-states the LB** (v4:
  val 0.98 vs LB 0.96); since the cross-encoder, the gap shrank (val ≈ 0.989–0.990 vs LB 0.9859).
- Remaining val loss (before India dense fix): India 0.018 = ~0.012 unretrieved + ~0.006 scoring; US 0.011 =
  ~0.0024 unretrieved (0.75% of true pairs) + ~0.009 scoring/decision. Precision ≈ 0.9986–0.9988, recall 0.950–0.968.
- If US/India test ≈ val, the LB implies **France ≈ 0.964** (inference, unmeasured). Could be US/India test being
  harder than val instead — we don't know; tell us how to find out cheaply (we have **12 LB submissions** left).
- Candidate set in `candidate_pairs.tsv`: **~21.0 per S1** (teammate's alternative pipeline: 65.8).

## 3. Production path of v13_k3 (read these; line numbers at this commit)
1. **Normalise + index** — `normalize.py`, `prep.py`, `terms.py` (Indic romanisation via `unicodedata` names,
   consonant skeletons, rare-term keys).
2. **Blocking** — `block.py` (numba budgeted rare-term retrieval, forward + reverse), `candgen.py`, `candwide.py`
   (~117 cand/S1 wide pool + cheap cross-fitted LightGBM), `select_cand.py` → "c2" ≈ 18–20 cand/S1.
3. **Stage 1 + Stage 2** — `stack.py` (93 pair features `features.py`; stage 1 cross-fitted over folds 1–4; stage-2
   164 features incl. group/competition/anchor/number-relation/rival-cluster; isotonic calibration). Models:
   joint (`s2xP3`, used for France), experts `s2xUS_P2`, `s2xIN_P2`. **Self-training**: `stack.py` `PSEUDO` +
   `add_pseudo()` (test pairs with p ≥ 0.98 & best claimant → positive; p < 0.02 "explained" → negative), 2 rounds.
4. **Test inference** — `infer2.py` → dumps (`row, tid, src, p` for p ≥ 0.02 + all candidates `c_*`).
5. **Cross-encoders** — `xenc.py` (CE1–CE4, e5-small, laptop GPU; `cmd_blend` :271 = logistic blend of logit(p2)
   and CE logits fitted **on the same val band it is evaluated on**), `export_pairs.py` / `remote_ce.py` /
   `import_ce.py` (CE5 = e5-large, H200). Band = pairs with p2 ≥ 0.02 ("all pairs" band, ~4.4/S1).
6. **Stage 3** — `stage3.py` (`group_stats` :36, `features` :76, `main` :92): LightGBM on logit(p2) + 5 CE logits +
   entity/target group statistics; 5-fold cross-fit over val entities; refit on all val → test.
7. **India dense rescue** — `bienc_export.py`, `remote_bienc.py` (e5-large bi-encoder fine-tuned with in-batch
   InfoNCE on 1.19M India train matches), `remote_dense.py` (embedding + FAISS IVF `knn`), `dense_newpairs.py`,
   `dense_merge.py` (`context` :41, `main` :53: cross-fitted LightGBM over CE5 logit, dense sim/rank, entity
   context → p for new pairs; merged into India dump; new pairs added to candidates).
8. **Decision** — `assemble.py` (`main` :42): concat per-country dumps; **mutual exclusivity** :61–65 (each target
   to its highest-p claimant); `decide.decide_all` (`decide.py` :31, expected-F0.5 prefix rule with abstention)
   with per-country `PARAMS` `US:1.1:1.0,India:1.1:1.0,France:1.1:4.0`.
   Drivers: `code/scripts/run_v13.sh` (and `run_seq2.sh`, `run_v12.sh`, `run_ce4.sh` for the parts before).

In flight right now: `run_v16.sh` (same dense rescue for US + France; French retriever trained partly on our own
confident French predictions), CE6 (second self-training round of e5-large).

## 4. What we want from you — go after these hard
Rank everything by **expected LB gain per minute**, with evidence (`file:line`) and the exact fix.

**A. Bugs and leaks that inflate validation or silently lose LB points.** Suspects we have not fully cleared:
1. `xenc.py cmd_blend` fits and reports on the same val pairs; `stage3.py` refits on all val then scores test —
   any place where a val number we trust is in-sample?
2. `stage3.py` group features: val groups are built from val-npz pairs with p2 ≥ 0.02; test groups from dump pairs
   with p ≥ 0.02 — are they really the same population (e.g. exclusivity applied before/after, pairs below 0.02,
   dense-added pairs missing from stage-3 groups)? Any train/test feature mismatch?
3. `dense_merge.py`: context features use stage-3 OOF p on val but full-model p on test; new pairs never went
   through stage 3; are new-pair probabilities on the same scale as existing ones when they compete in
   `decide_all` / exclusivity?
4. `assemble.py`: exclusivity across all countries' dumps; order of exclusivity vs decision; `MIN_P` = 0.02 floor;
   any id/row mapping (`row` = S1 cache row, `tid` = numeric id) that could mis-assign or drop pairs; duplicate
   candidates between lexical and dense sets.
5. Self-training (`stack.py add_pseudo`, `xenc.py PSEUDO_DUMP`): confirmation bias or label leakage that the LB
   would punish but validation cannot see.
6. `decide.py decide_all`: is the expected-F0.5 rule correct for this metric (5c/(T+4P) per entity; empty = 1 only
   for singletons)? Are `t_scale`/`abstain` doing what we think?
7. Anything in normalisation/blocking (`normalize.py`, `terms.py`, `block.py`) that drops or mangles records
   systematically (e.g. empty fields, French/Indian specifics) — measurable as recall loss.

**B. Easy wins we are leaving on the table** (decision layer, per-country settings, thresholds, K for dense
additions, France's cautious `abstain=4.0` which came from an old leave-one-country-out test, boosts, anything).

**C. The val→LB gap.** Given everything above, where do you think the missing ~0.004–0.005 vs our validation
actually is (France? US/India test harder? decoys?), and **which cheap LB probes** (we have 12) would tell us?

**D. Candidate generation for the final review.** Is ~21 candidates/S1 with the funnel ~117 → 20 → ~7 → 4.4
defensible and honestly reported? Any way to shrink it without losing recall, in < 1 hour?

**E. Disqualification / reproducibility risks** in the code as it stands (rules in `docs/problem_statement.pdf`).

**Output**: `astra/REVIEW_1.md` — top section "**Do these in the next 2 hours**" (max 5 items, each with expected
gain, confidence, minutes, exact fix), then the full ranked findings. Be blunt. We would rather hear "this is
wrong" than "consider reviewing".
