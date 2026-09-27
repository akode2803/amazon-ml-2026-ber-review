# Amazon ML Challenge 2026 — Business Entity Resolution: our journey

*Status as of 27 Sep 2026, ~15:45 IST. Final deadline: **23:00 IST today**.*

---

## 0. The one-page summary

| | |
|---|---|
| **Task** | For every business record in Source 1 (the clean reference list), find all records in Source 2 and Source 3 that describe the same real business. |
| **Metric** | **Macro F0.5** per Source-1 entity (precision counts twice as much as recall), averaged over all entities, including those with no match. |
| **Our aim** | Top 25 on the leaderboard. The 25th-place score is currently **0.9909**. |
| **Where we are** | Best public score **0.981866** (version v12s3), rank ~603. Gap to top 25: **~0.009**. |
| **How far we came** | From **0.957** (the first team submission) to **0.981866** today, mostly in the last 20 hours. |
| **What moved the score most** | (1) Self-training on unlabeled test data, (2) a multilingual text model (cross-encoder) that judges pairs from their raw text, (3) a larger version of that model on a rented H200 GPU, (4) a "stage 3" decision model. |
| **Biggest remaining problem** | India: ~3.2% of true matches are never *retrieved* at all, mostly because one source writes the business name in English and another in Hindi/Bengali/Tamil script. |
| **What we are doing about it** | A fine-tuned multilingual retrieval model that finds these cross-script matches: it recovers **78%** of the missed India matches with fewer than 2 extra candidates per entity. |
| **Other constraint that counts** | Final ranking also rewards a **small candidate set per entity** and a **scalable** blocking method. Ours: **~20 candidates per entity** (teammate's version: 65.8), with no all-pairs comparison anywhere. |

---

## 1. The problem and the data

### 1.1 What we are given

Three independent sources of business records. Each record has only four fields:

| Field | Example | Notes |
|---|---|---|
| `entity_id` | `S2-187020300` | Prefix tells the source (S1/S2/S3). IDs are random — no leakage (we checked: row order and ID values carry no signal). |
| `business_name` | `Orelee's Barbershop`, `राम मार्केटिंग प्राइवेट लिमिटेड` | Abbreviations, legal suffixes, typos, transliteration, **native Indian scripts**, sometimes a totally replaced name. |
| `business_address` | `1795 Westchester Drive, High Point, NC` | Partial, reordered, abbreviated, landmark-based ("Near SBI ATM"), sometimes empty. |
| `country` | `US`, `India`, `France` | Training has US and India only. **France appears only in the test set** (zero-shot). |

Source 1 is deduplicated: each business appears once. Sources 2 and 3 can hold several records of the same business, and many records that match nothing.

### 1.2 Sizes

| | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| **Train** | 2,206,821 | 5,034,616 | 5,285,603 |
| **Test** | 1,732,544 (France 259,452 · India 809,986 · US 663,106) | 4,887,273 | 5,082,316 |

Training ground truth: 7.64 million match links. 5.58% of Source-1 entities have **no** match (singletons). An entity has 3.46 matches on average (about 1.7 in S2 and 1.8 in S3).

### 1.3 How we are scored

- For each Source-1 entity: F0.5 = 1.25·P·R / (0.25·P + R), where P and R are the precision and recall of *its* predicted list.
- A singleton scores 1.0 if we predict nothing, 0.0 if we predict anything.
- The leaderboard score is the average over all entities. The public leaderboard uses part of the test set; **the final ranking uses the other (private) part**.
- `matching_results.tsv` is the scored file. `candidate_pairs.tsv` is **not** scored on the leaderboard but is **reviewed for the final ranking**: smaller candidate sets per entity rank higher, and blocking must scale (no comparing every record with every other).

### 1.4 Rules that shape our choices

- **No external data** (no business registries, geocoding, lookup APIs). Only the provided files.
- **Models must be MIT/Apache-2.0 licensed and ≤ 8 billion parameters.**
- The final zip must regenerate both output files from the code inside it.
- Our models: `intfloat/multilingual-e5-small` (118M) and `intfloat/multilingual-e5-large` (560M), both **MIT**. A rented GPU is compute only, not data — allowed.

### 1.5 Hidden traps in the data (found along the way)

| Trap | Evidence | Why it matters |
|---|---|---|
| **More distractors in test** | Train: ~27% of S2/S3 records match nothing. Test: estimated **~41%** (2.82 vs 2.28 target records per S1). | Models tuned on training data accept too many near-miss look-alikes on test. |
| **"Decoy siblings"** | Same street, shifted building number, branch words — ~3× more frequent among test candidates. | Classic false positives; F0.5 punishes them hard. |
| **France is zero-shot** | No French training labels; French addresses have almost no postcodes (0.3%). | Anything learned from US/India must transfer. |
| **Cross-script India records** | 36% of India training match pairs have native script on at least one side ("New Products Pvt Ltd" ↔ "न्यू प्रोडक्ट्स प्राइवेट लिमिटेड"). | Keyword-based retrieval cannot connect them → true matches are never even considered. |

---

## 2. Constraints we worked under

| Constraint | Detail | Consequence |
|---|---|---|
| **Time** | ~2.5 days in total; final deadline 23:00 IST on 27 Sep. | Every experiment must earn its time. |
| **Laptop** | 20 CPU cores, 31 GB RAM, RTX 5060 laptop GPU (8 GB). | Most of the pipeline is CPU-bound; memory is the real bottleneck. |
| **Memory** | A single stage-2 build used up to ~16 GB. Two in parallel exhausted RAM. | Claude Code's safety mechanism stopped our jobs twice. We moved to strictly sequential queues. |
| **Submissions** | Limited uploads per day. | Each submission had to test one clear hypothesis. |
| **No labels for France** | France cannot be measured directly. | We used "leave-one-country-out" experiments (train on US, test on India) as a stand-in. |
| **Candidate-set size counts** | Reviewed in the final ranking. | We keep ~20 candidates per entity and add new ones only when validation proves they pay. |
| **Rented GPU budget** | RunPod H200 NVL at ~$3.80/hour, prepaid credit (~$25). | GPU jobs are queued back to back; the pod is stopped when done. |

---

## 3. The pipeline, stage by stage (current version)

```
 raw records
     │  1. normalise (Unicode folding, Indic → Latin romanisation, phonetic skeletons, field parsing)
     ▼
 2. BLOCKING  — rare-term inverted index, forward (S1→S2/S3) and reverse (S2/S3→S1)   ~117 candidates / entity
     │         + cheap LightGBM re-ranker keeps the best                              ~20 candidates / entity  ← candidate_pairs.tsv
     ▼
 3. STAGE 1   — LightGBM pair model, 93 similarity features, cross-fitted            keeps ~7 / entity
     ▼
 4. STAGE 2   — stacked LightGBM, 164 features incl. competition between candidates  (per-country experts + joint model for France)
     │           + SELF-TRAINING on pseudo-labelled test pairs (2 rounds)
     ▼
 5. CROSS-ENCODERS — multilingual text models read both records and judge "same business?"
     │           e5-small ×4 (laptop) + e5-large (H200), on the ~4.4 pairs / entity with p2 ≥ 0.02
     ▼
 6. STAGE 3   — LightGBM decision model over all scores + context (rank in entity, margin to runner-up, rival claims)
     ▼
 7. DECISION  — one target record belongs to at most one entity (mutual exclusivity);
                expected-F0.5 rule decides how many candidates to accept per entity (can abstain)
     ▼
 matching_results.tsv
```

### 3.1 Normalisation (`normalize.py`, `prep.py`)
- Unicode folding (accents stripped), lower-casing, punctuation → space.
- **Indic romanisation** built from Unicode character *names* (an algorithm over Python's `unicodedata`, not a lookup table): "राम" → "raama".
- **Phonetic consonant skeletons** so spelling variants share a key ("Marketing"/"maarketinga" → "MRKTN").
- Field parsing: name tokens, legal suffixes (Pvt/Ltd/LLC/SARL…), street-type words (US, French, Indian), numbers, geography words.

### 3.2 Blocking / candidate generation (`terms.py`, `block.py`, `candgen.py`, `candwide.py`, `select_cand.py`)
- Each record is indexed under its **rarest** terms (names, address words, numbers, skeletons, compound keys). A query only visits records sharing a rare term, within a fixed budget → cost grows with the number of candidates, not with the dataset.
- Forward and reverse retrieval are unioned (reverse catches cases where the S2 record is the "rare" side).
- A cheap cross-fitted LightGBM re-ranks the ~117-candidate pool and keeps ~20. Validation recall is 98.29% at 18 candidates; keeping all 117 would only add +0.07 points of recall.

### 3.3 Stage 1 — pair model (`features.py`, `stack.py`)
93 features per candidate pair: token overlap per field with IDF weights and "contradiction" signals (a rare token on only one side), rapidfuzz string similarities (token-set, Jaro-Winkler…), retrieval ranks, and record-signature features (how many records share this exact address/name). LightGBM, cross-fitted over 4 folds so every training pair is scored by a model that never saw it.

### 3.4 Stage 2 — stacked model (`stack.py`)
164 features: the stage-1 score plus **relational** signals — the pair's rank among the entity's candidates, best/second-best scores, how many entities claim the same target, similarity to the entity's other strong candidates, building-number relations (typo vs neighbouring building), rival clusters. Trained per country ("experts" for US and India) plus one joint model used for France.

### 3.5 Self-training on test data (`stack.py PSEUDO=...`)
The model labels the test pairs it is very sure about (p ≥ 0.98 → positive; very low p → negative, but only when "explained" by another confident match), then retrains with them. This adapts the model to the test set's own distribution (extra distractors, France) without any external data and without test labels.

### 3.6 Cross-encoders (`xenc.py`, `remote_ce.py`)
A transformer reads *both records as text* and outputs a "same business?" probability. It understands abbreviations, word order, transliteration and French words far better than hand-built features.
- CE1–CE4: `multilingual-e5-small`, trained on the laptop GPU; CE2–CE4 each add pseudo-labelled test pairs (self-training for the text model). Holdout AUC 0.99756 → 0.99852 → 0.99898 → 0.99929.
- CE5: `multilingual-e5-large` on the rented H200 — holdout AUC **0.99952**.
- They score only the pairs stage 2 keeps (p2 ≥ 0.02, ~4.4 per entity); a blend then combines them with stage 2.

### 3.7 Stage 3 (`stage3.py`)
A small LightGBM that sees the stage-2 score, every cross-encoder score, and context: rank within the entity, top/second score, margin, and how strongly other entities claim the same target. Cross-fitted over validation entities (honest estimate), then refit for test.

### 3.8 Decision (`decide.py`, `assemble.py`)
- **Mutual exclusivity:** a Source-2/3 record goes to the entity most confident about it (the ground truth is many-to-one).
- **Expected-F0.5 rule:** for each entity, accept the top-k candidates that maximise expected F0.5; abstaining (predicting nothing) is a first-class option — it earns 1.0 on singletons.
- Per-country operating points (France slightly more cautious).

---

## 4. The journey — what we tried, in order, and what the leaderboard said

| Version | What changed | Validation | **Leaderboard** |
|---|---|---|---|
| v1 | First team pipeline (lexical blocking + LightGBM + decision rule) | 0.964 | **0.957** |
| v3 | Wider re-ranked candidates, stage-2 stack, per-country experts | 0.981 | **0.958547** |
| v4 | Stage-2 number-relation + rival-cluster features | 0.9825 | **0.95949** |
| (teammate c5) | Separate pipeline: 66 candidates/entity, char 3-grams, one LightGBM | 0.970 | **0.964172** |
| v5b | **Self-training for France** (cleaner "explained" negatives) | — | **0.96101** |
| v6ce | + self-training US/India + **cross-encoder** (e5-small) | +0.003–0.005 | **0.976768** ← biggest jump |
| v8all | + 2nd self-training round, cross-encoder on **all** kept pairs | — | **0.978831** |
| v9b | + CE2, cross-encoder weight × 1.5 | — | **0.980723** |
| v10b | + CE3 (self-trained on all countries' test pairs) | — | **0.980926** |
| **v12s3** | + **e5-large cross-encoder (H200)** + **stage 3** | US 0.9895 / India 0.9818 | **0.981866** ← current best |

### Experiments that did **not** work (and why that was useful)

| Idea | Result | Lesson |
|---|---|---|
| Monotone + heavily regularised LightGBM ("robust" model) | Worse on the unseen country (0.87499 vs 0.89027) | Robustness had to come from data and text models, not constraints. |
| EM prior-shift correction for the extra distractors | Leaderboard −0.0005 | Recalibrating probabilities alone does not fix the shift. |
| Widening candidates from 18 to 117 per entity | +0.07 recall points | Missed matches were not "just below the cut"; wider lists only add noise. |
| Averaging the joint model with the country experts | +0.0002 validation | Too small to spend a submission on. |
| Re-tuning the decision rule on cross-encoder probabilities | 0 | The rule was already optimal; remaining errors are in the scores. |
| Cross-encoder rescue of pairs scored 0.003–0.02 | Only 0.1–0.3% of true pairs live there | Not worth scoring a band where 1% of pairs are positive. |
| Off-the-shelf multilingual embeddings for retrieval | Recovered only 23% of India's missed matches, +3.2 candidates/entity | Needed fine-tuning (see §7.1). |
| Source-balance features in stage 3 | No change | Stage 3 already captures it. |
| CLM-v0.1-8B (agent re-ranker) | Rejected without running | Wrong task, English-only, ~8.2B params (rule risk), ~6 h to score. |

---

## 5. Problems we hit and how we solved them

| # | Problem | How we found it | How we overcame it |
|---|---|---|---|
| 1 | **Validation said 0.98, leaderboard said 0.96** | v1→v3 gained +0.017 on validation but only ~+0.002 on the leaderboard. | Diagnosed a distribution shift (41% vs 27% distractors, France zero-shot). Fixed with self-training on test data and text models that generalise. The gap is now much smaller (validation ~0.989 vs leaderboard 0.982). |
| 2 | **France has no labels** | — | Used "train on US, test on India" as a stand-in to choose methods (self-training: +0.005 → +0.014 → +0.022 on that stand-in), then confirmed on the leaderboard (+0.0015 from France alone). |
| 3 | **Laptop ran out of memory** (jobs killed twice) | Claude Code's low-memory safety stop. | Rewrote stage 2 to spill each country's matrix to disk and train via LightGBM `Dataset.subset` (peak memory roughly halved); ran heavy jobs strictly one at a time. |
| 4 | **A silent race condition in blocking** | The retrieval kernel used thread-id % 10 as scratch space on a 20-core machine. | Scratch rows sized to the real thread count (and `NUMBA_NUM_THREADS=10` when running the teammate's code). |
| 5 | **The teammate's pipeline scored higher than ours (0.964 vs 0.961)** | Leaderboard. | Compared designs; concluded our stage-2 relational features were hurt by the test shift. Instead of switching pipelines, we added the cross-encoder — which jumped us to 0.977. |
| 6 | **Cross-encoder gains invisible on validation** | Validation +0.004, leaderboard +0.016. | Started trusting leaderboard evidence for text-model changes (e.g. boosting the cross-encoder weight: validation −0.0001, leaderboard +0.0019). |
| 7 | **Small models saturating** | CE3 added only +0.0002. | Rented an H200 for `e5-large` (560M) → holdout AUC 0.99952, leaderboard +0.00094 with stage 3. |
| 8 | **Rented GPU setup** | SSH key not installed at pod creation; pip blocked by PEP 668; pod limited to ~13.6 CPU cores despite showing 128. | Added the key via the web terminal; `--break-system-packages` in a throw-away container; capped tokenizer/FAISS threads. |
| 9 | **India matches never retrieved** | Error breakdown (§6). | Fine-tuned multilingual retrieval (§7.1). |

---

## 6. Where the remaining points are lost (validation error breakdown)

| | India | US |
|---|---|---|
| Total remaining loss (1 − F0.5) | **0.018** | **0.011** |
| True matches **never retrieved** | **~0.012 (two-thirds!)** — 3.16% of true pairs | ~0.002 — 0.75% of true pairs |
| Scoring / decision errors | ~0.006 | ~0.009 |
| Precision | 0.9988 | 0.9986 |
| Recall | 0.950 | 0.968 |

Precision is already excellent. **Almost all remaining loss is recall** — and for India, mostly matches our blocking never finds.

India's missed matches show why: one source writes the name in English, the other in a native script:

- "New Products Private Limited" ↔ "न्यू प्रोडक्ट्स प्राइवेट लिमिटेड" (Hindi script)
- "Jain Marketing" ↔ "জৈন মার্কেটিং" (Bengali)
- "White Infrastructure Private Limited" ↔ "வொயிட் இன்ஃப்ராஸ்ட்ரக்சர் பிரைவேட் லிமிடெட்" (Tamil)

Our romaniser turns these into forms like "nyuu prodakts…", which never share a rare keyword with "new products".

---

## 7. What we are doing right now (and why)

### 7.1 Fine-tuned multilingual retrieval for India ⭐ (biggest lever)
- `multilingual-e5-large` fine-tuned on 1.19M India training matches (887k involve native script), with in-batch contrastive learning: every record must pick its own match out of the batch.
- Search with a **FAISS IVF approximate index** — each query scans ~1–1.5% of records, so it scales (no all-pairs comparison).
- Result on validation (India, 19,346 missed true matches):

| Retriever | top-2 | top-3 | top-5 |
|---|---|---|---|
| Off-the-shelf e5-large | — | — | 23% recovered, +3.2 candidates/entity |
| **Fine-tuned** | **58%**, +0.68/entity | **78%**, +1.72/entity | 91%, +4.74/entity |

- New pairs are scored by the large cross-encoder and a small cross-fitted model, then merged. We choose K (2, 3 or 5) on validation, balancing the score gain against candidate-set size. → **v13**.

### 7.2 Transitive cross-encoder ("anchor" check) → **v14**
An entity has ~3.5 records across S2/S3. Once one is matched confidently (the anchor), an uncertain candidate is also compared with **that record**, not just with the S1 record. Two views of the same business are often closer to each other than to S1 (e.g. both in English while S1 differs). 9.7M anchor pairs are scored on the H200 and added as stage-3 features. No new candidates.

### 7.3 Second self-training round of the large cross-encoder (CE6) → **v15**
Continue training `e5-large` on 2M fresh training pairs + 1.5M pseudo-labelled test pairs from v12s3. Every self-training round so far has improved the model.

### 7.4 Timeline for the rest of the day

| Time (IST) | Milestone |
|---|---|
| ~16:05 | v13 (India retrieval) ready → submission |
| ~16:30 | v14 (+ transitive cross-encoder) ready → submission |
| ~17:45 | v15 (+ CE6) ready → submission |
| ~20:30 | Best combination → submission; stop the rented GPU |
| 21:00–22:00 | Final submission package (code, README, documentation) |
| ~22:15 | Final upload (reserve slot kept) |

---

## 8. Current issues and how we plan to handle them

| Issue | Plan |
|---|---|
| **Gap to top 25 (~0.009), and the cut-off keeps rising** | Stack the three levers above; each targets a different error type, so gains should add. Realistic outcome 0.985–0.989; top 25 needs everything to land. |
| **Candidate-set size is judged** | Report the honest set (~20/entity). Add India retrieval pairs only for the small K that validation justifies (≈ +0.3 to +0.8 per entity overall). Document the funnel: ~117 → 20 → ~7 → 4.4. |
| **Validation under-states text-model gains** | Use the leaderboard (6 submissions left) as the judge for text-model changes; validation for everything else. |
| **Private leaderboard ≠ public** | Prefer changes with clear gains (> 0.001); self-training used the whole test set, so it helps both parts. |
| **Reproducibility of a multi-machine pipeline** | The final zip will contain every script, the exact run order, model names and licences, and which steps need a ≥ 24 GB GPU. |
| **Rented GPU cost / credit running out mid-job** | Jobs queued back to back; results saved to the persistent volume; the pod is stopped as soon as the last job ends. |
| **Laptop memory** | Keep heavy jobs strictly sequential. |

---

## 9. Scalability and candidate generation (for the final review)

| Stage | Candidates per S1 entity | Method | Scales because… |
|---|---|---|---|
| Rare-term retrieval | ~117 | Inverted index, per-record budget of rarest terms | Work ∝ candidates, not dataset size |
| Cheap re-ranker cut → **candidate_pairs.tsv** | **~20.3** | Cross-fitted LightGBM | Linear in candidates |
| Stage-1 filter | ~7 | LightGBM | — |
| Cross-encoder input | ~4.4 | Transformer on survivors only | The expensive model sees a tiny fraction |
| India dense retrieval (new) | +0.7 to +1.7, India entities only | Fine-tuned embeddings + **FAISS IVF** | Each query scans ~1–1.5% of records |

For comparison, the teammate's pipeline reported **65.8** candidates per entity.

---

## 10. Where things live

| Path | What |
|---|---|
| `Amazon_ML\ber\src\` | All our code (normalisation → blocking → stages 1–3 → cross-encoders → decision) |
| `Amazon_ML\ber2\` | Working caches, models, logs (`ber2\logs`), and every output folder (`ber2\output_v*`) |
| `Amazon_ML\ber2\output_v12s3\` | Current best submission (0.981866) |
| `Amazon_ML\atharwa\` | Teammate's c5 submission (0.964172), extracted for comparison (not used) |
| RunPod `/workspace/ce/` | Large cross-encoder, retrieval model, scores |

Key scripts: `stack.py` (stages 1–2 + self-training), `xenc.py` (small cross-encoders + blending), `remote_ce.py` / `export_pairs.py` / `import_ce.py` (large cross-encoder on the GPU pod), `stage3.py`, `remote_bienc.py` / `dense_*.py` (India retrieval), `anchor_export.py` (transitive cross-encoder), `assemble.py` (final decision + TSVs).

---

## 11. Glossary

- **Blocking / candidate generation:** cheaply shortlisting which records *might* match, so the expensive model only compares a few pairs.
- **Cross-fitting:** training on some folds and predicting others, so no record is scored by a model that saw it.
- **Cross-encoder:** a transformer that reads two records together and outputs a match probability (accurate, expensive).
- **Bi-encoder / embedding retrieval:** a transformer that turns each record into a vector once; similar records get nearby vectors (cheap to search at scale).
- **FAISS IVF:** an approximate nearest-neighbour index that clusters vectors and searches only a few clusters per query.
- **Self-training / pseudo-labels:** letting the model label test data it is very sure about, then retraining on those labels.
- **Macro F0.5:** an F-score weighting precision twice as much as recall, computed per entity and averaged.
- **Leave-one-country-out (LOCO):** train on one country, evaluate on another — our stand-in for France.
