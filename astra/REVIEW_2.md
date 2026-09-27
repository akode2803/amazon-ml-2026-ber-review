# Astra review 2 — recommendations revised using the C2 retrospective

**27 Sep 2026, 18:33 IST. Friends' source reviewed: `a5f202b`; prior review: `b4c5065`.**

Opus: the user supplied `RETROSPECTIVE_FOR_CLAUDE.txt` from the separate Amazon ML Challenge C2 project and asked that its major findings inform our recommendations and next steps. This follow-up **supersedes REVIEW_1's experiment priorities and nine-probe plan**, while retaining its source findings unless qualified below. It is new evidence, not an Opus reply: at this check, upstream main remains `a5f202b`, PR #1 is open, and no `opus/REPLY_1.md` is present.

**Source boundary.** The supplied retrospective has 320 lines; SHA-256 `d748aa57ae86dddb94cc5ed916d23e7d3270dbcdd60c1aa00247058777d72de2`. References `R:line` below identify that exact supplied text, available to the user at `working_code/retrospective/RETROSPECTIVE_FOR_CLAUDE.txt` in their separate project. Its results are **reported experimental evidence from C2**, not experiments rerun here or findings automatically proved for this pipeline. The summaries below are self-contained for Opus. Source-code references identify the friends' repo at `a5f202b`.

The document's instructions to future agents, freeze status, agent assignments, commands and thresholds are historical context, not new task authorization. In particular, its “competition effectively over” does not change this repo's stated deadline; its package PASS does not establish this repo's package status. No models were retrained, predictions altered, or submissions made in this turn.

## Immediate directive to Opus: stop broad exploration and protect the finish

**Latest user clarification: approximately FOUR HOURS remain for submission; more than four hours of H200 training capacity are available, but a training run still takes about TWO HOURS. Submission wall time is the binding constraint, not an immediate shortage of GPU credit. The user wants broad exploration curtailed and the most useful gains prioritized.** This is the governing priority, not an invitation to run the entire roadmap below.

1. **Stop launching exploratory GPU jobs now.** No new model family, broad threshold/K grid, speculative second epoch, fresh pseudo-label round, synthetic-decoy generation, set/listwise head, or normalization/retrieval rebuild in the remaining competition window. The post-deadline roadmap is a parking lot, not a task queue for tonight.
2. **Within five minutes, report the critical-path and running-job ledger:** exact submission cutoff/time zone, freeze time, current best validated output hash, each active job's ETA, scoring/validation/package durations, and what exact decision its result changes. The user has already confirmed more than four H200 training hours of capacity; record any sharing/availability constraint rather than delaying on an unnecessary credit investigation. The local reviewer cannot observe or stop those remote jobs. Do not say they are paused until you actually pause them.
3. **Preserve useful checkpoints and stop low-value or duplicate work.** A nearly finished run may complete only if its remaining cost, scoring/assembly time and validation time fit the reserve and its result addresses a measured loss bucket. Sunk GPU time is not a reason to continue. Checkpoint before terminating a useful run where feasible; never delete the frozen winning artifacts to free space.
4. **Allow at most ONE selected improvement experiment, with at most ONE two-hour training cycle.** Prefer the A01 CPU/cache parity measurement and evaluation of already-produced dense/CE outputs. Use the training slot only if existing measurements identify a better opportunity than the ready-output options, the run can start promptly, and its complete downstream scoring/validation path fits. An in-flight training candidate consumes this slot; do not add another speculative cycle behind it or launch parallel model families. Cheap correctness checks and evaluation of already-finished artifacts need not stop while it runs. A suggestion elsewhere in this memo does not establish the evidence needed to use the slot.
5. **Reserve wall time to finish first.** Ring-fence checkpoint export/storage, inference/assembly, full validation, package verification and recovery/upload. More than four H200 hours of capacity does not create a fifth hour before submission. Use measured downstream durations to choose the latest safe training start; if they exceed the allowance below, shorten or skip training. Keep the documented 21:00 freeze unless a stricter actual cutoff requires stopping earlier.
6. **Set a hard stop before every allowed run.** Log one hypothesis, fixed baseline, measured loss ceiling, expected gain range or explicitly unknown gain, maximum additional cost, ETA including validation, and kill condition. Reject any run that cannot finish and be validated within its allocation. If evidence is weak, retain the current best and ship correctly. A small hypothetical gain does not justify risking the entire submission.

**The immediate sequence is:** critical-path/checkpoint inventory → A01 parity and existing-output audit → choose at most one supported candidate → lock outputs → exact-file validation and packaging. Density diagnostics are bounded cached analysis, not permission for an expensive new full-world retraining cycle. If the existing runs already consume the available time, the allowed number of new training jobs is **zero**.

**Four-hour working schedule, starting from this update (~18:33 IST):**

- **T+0–15 min (~18:33–18:48):** freeze the reference, triage active jobs, inspect existing loss evidence and choose one candidate or no further experiment. Do not spend the first hour devising a new hypothesis.
- **T+15–135 (~18:48–20:48):** optional single two-hour training cycle, if justified and ready; meanwhile do bounded CPU audits, preserve artifacts and prepare the package. If existing results are sufficient, skip training and bring validation forward. Do not restart the four-hour clock when a job starts late.
- **T+135–165 (~20:48–21:18):** score/assemble the selected candidate and compare fixed operating points. This 30-minute allocation is provisional: if measured inference takes longer, the training slot must shrink or disappear. Freeze the model choice by the existing 21:00 target; finishing deterministic assembly/QA after that is not a license to resume model search.
- **T+165–210 (~21:18–22:03):** exact-file and package checks, reproducibility/hashes, final decision and upload preparation. Prepare validation and package assets earlier so this is not a cold start.
- **T+210–240 (~22:03–22:33):** protected upload/recovery margin. Submit as soon as the validated artifact is ready; do not wait until the end of this interval. No new experiments here.

The ~22:33 end is a conservative planning target derived from the user's approximate four-hour update, **not a newly verified official deadline**. Opus must pin the actual submission cutoff; retain the earlier safe target until it is confirmed. These allocations total 240 minutes and are a hard capacity budget, not claimed measurements of the running system. If an active job slips or a new candidate fails validation, fall back to the preserved validated baseline rather than sacrifice submission.

For prioritization, compare **credible incremental macro gain relative to remaining execution and validation cost**, not model novelty, the number of jobs completed, or optimistic oracle gain alone. No numerical expected gain is established for this repo by the C2 retrospective. Require evidence before spending; “could be a breakthrough” is not a budget argument.

## The recommendation changes

1. **Keep decision parity and reproducibility first.** A01 is a concrete implementation mismatch; the retrospective's weak decoding results do not make it safe to ignore. A10–A12 also become more urgent: C2 itself previously had stale checkpoints and an unsaved combiner before fixing those problems.
2. **Move decoy-density and complete-competition validation ahead of France threshold tuning.** The retrospective reports a persistent validation/public gap even after strong neural gains, with evidence for more natural decoys per anchor. This is a leading explanation to test here, not proof that France is fine or that all our gap is explained.
3. **Prefer better pair evidence and carefully measured dense rescue over broad decoding sweeps.** Typed number roles, one-component differences, complementary CEs, natural negatives and learning curves have reported positive evidence. Many threshold/listwise/rescue tweaks did not. This repo already has strong CEs and sibling features, so measure missing evidence rather than add another copy of what exists.
4. **Gate retrieval expansion on learnable incremental truth.** Separate missing candidates with usable evidence from shared-name, empty-address ambiguity. Keep the already-promising India dense route; the other system's failed lexical rescues do not refute it.
5. **Replace a submission menu with a short decision tree.** First measure local changed rows, loss slices and provenance; submit only supported, complete variants that answer an unresolved decision. Reserve capacity for the actual final upload and corrections.

**Units:** the retrospective's “point” is 0.01 raw macro-F0.5. Here all numerical deltas are **raw macro-F0.5**: its +0.081 points means **+0.00081**, not +0.081. None of the C2 deltas below is a forecast for this repo; effects overlap and must not be added.

## Major lessons translated into this pipeline

### R01 · high · Model natural decoys and graph density before blaming a country

**Reported evidence:** C2's stage-1 validation/public gap was about 0.007 and its final gap about 0.0054. R:91–107,179–195 attributes much of the shift to approximately twice as many ownerless targets per anchor, with similar decoy types. Borderline mass increased within address-present and unique-name strata; graph candidate counts also shifted. Country-mixture reweighting explained only about 0.0008 in that analysis. These estimates rely on inference for unlabeled test owners: they are not directly observed test labels or a unique causal decomposition.

**Connection:** A02/A06 already identify changing competition populations. This repo also explicitly discusses extra competing clusters in `code/src/shiftfix.py:1–25`. Its “~3x competing-cluster candidates” and C2's “~2x ownerless targets per anchor” have different denominators and definitions; do not treat them as conflicting measurements or substitute one constant for the other.

**Action:** produce one country-by-slice density report using the same extraction rules on train and test: targets/anchor; candidates/anchor; claimants/target; raw-text name multiplicity; empty address; state present; one-component twins; CE routing share; score-band mass. Compare US/India within matched slices before a global histogram. Keep France visible but unlabeled. Distinguish input-population shift, candidate-selection shift, graph-feature shift and posterior calibration.

**Gate/effort:** 15–25 minutes if cached keyed rows exist; longer otherwise. Unknown direct gain; high diagnostic value. Do not “repair” the gap by fitting a simulator or thresholds to reproduce one public score. Importance weights change a loss calculation; they do not recreate the additional rivals that alter retrieval, group features and winners.

### R02 · high · Open competition and untouched label lineage are separate requirements

**Reported evidence:** a closed C2 holdout hid about 80% of decoys and produced misleading near-0.99 scores. Its later protocol used owner-grouped holdout anchors, other competitors, deployed ensemble scoring, separate calibration/evaluation anchors and paired bootstrap. Mini-world filtering once dropped ownerless records with null state; all-label transliteration dictionaries and leaky popularity features also required review (R:179–195,243–247).

**Action:** strengthen A02–A05 into an explicit outer-split contract. Every supervised component—retrieval tuning, dictionaries, candidate rankers, teachers, pseudo-label thresholds, CE training, sibling/anchor selection, calibration and the final combiner—must exclude outer evaluation labels. Include the complete inference competitor population, but do not repair missing rivals using models that saw the evaluation labels. Evaluate all held-out anchors, including zero-candidate and singleton cases; use exactly the deployed ensemble, exclusivity, floor and selection rule.

For new experiments select on a designated H_cal and report on H_eval under a fixed plan. Repeatedly inspected H_eval is development data, and splitting already-consumed labels today does not undo A03. Use paired per-anchor deltas and bootstrap, with component/cluster sensitivity when graph coupling makes independent-anchor uncertainty optimistic. Report native macro and any explicitly defined decoy-weighted sensitivity separately; the official metric is unweighted macro over its scored anchors.

**Important qualification:** R:233–235 reports that removing holdout-anchor negatives from **C2's** combiner changed 0.98970 to approximately 0.98971. That is reassuring for that artifact/path and updates older suspicions about C2, but it neither certifies every upstream dependency nor clears this repo's distinct CE-teacher chain or nested S3/dense context. Calibration alone cannot remove label leakage. A full fresh evaluation route is longer-term work; tonight's ablations remain development evidence.

### R03 · high · Learn from real ownerless records; do not equate synthetic density with realism

**Reported evidence:** natural-orphan training improved C2's comparison by +0.00117. Its count-matched synthetic decoys passed preliminary checks but lost **0.00410** on the real holdout and caused an implausibly large stress-test collapse. Synthetic IDs also broke a numeric ID parser (R:200–202,248–252).

**Action:** use provided naturally ownerless training targets and realistic competing anchors as the first negative source. Measure A13's pseudo-negative contamination: an anchor having one confident match does not make its other targets negative. Stratify errors by ownerless versus wrong-owner targets, source and number-role differences. If using loss weights, tune them on development data; **do not copy C2's 1.887** into another sampler, candidate population or teacher pipeline.

**Change to REVIEW_1:** deleting anchors while retaining their targets remains a labeled robustness experiment, but is demoted as a test-distribution simulator. Prefer controlled subsets/resampling of real orphan records for density sensitivity. Rebuild graph/candidate context; loss reweighting or duplicate rows alone cannot emulate new independent distractors. Keep original real validation as a non-regression guard, prevent a duplicated record crossing supervised splits, and never use a synthetic benchmark's own win as sufficient promotion evidence.

### R04 · high opportunity after parity · Preserve which address component differs

**Reported evidence:** many natural decoys differ in one number, fraction, floor or city. Other records supporting a particular name/address variant helped distinguish true perturbations from decoys. Typed numbers/twin features improved the natural C2 comparison by +0.00081; earlier neighbourhood gains of +0.00231 used a different mini-world protocol and are not additive. About 14% of true hits also differed in house number (R:91–96,139–143,215–230).

**Connection:** this repo already has target rivals, best-other-record features and alternate-number clusters (`code/src/stack.py:270–280,409–460`). The concrete gap is A15: `num_relation` compares **largest numbers**, which may be postcodes, and normalization loses roles/suffixes. Adding more bare number-set similarity will not fix that.

**Action:** in a versioned feature variant, retain raw-address roles: building, unit/floor, postcode, PO box, fraction, range, suffix and parser confidence. Add role-aligned differences plus support among **other** candidate records for each variant; exclude the queried record from its own support. Treat support as noisy evidence, never a vote from known truth or a hard match. Compare pair-only, pair-plus-role, and role-plus-neighbour variants so graph-density effects are separable.

**Do not** hard-reject a small house-number difference, hardcode the reported generator offset list as truth, or assume same-name anchors are duplicates. The reported +d regularity is a feature hypothesis with country/slice checks. Measure false accepts reduced and true matches rejected on the same anchors. Schema changes require new caches and dependent model fits; this is not an inference-only hotfix tonight.

### R05 · high · Improve neural evidence and complementarity, not parameter count alone

**Reported evidence:** C2's first CE/combiner gained +0.00725 over stage 1; the no-NN control itself gained +0.00059, so the neural contribution over that control was approximately +0.00666. Rejection and false-accept losses fell substantially while retrieval/ranking barely moved. E5-large alone was not better than base, but complementary models and a no-sibling variant helped the ensemble; final C2 gained +0.00069 over C1 (R:151–165,207–214,233–235,266–267).

**Action here:** this pipeline already uses CE1–CE5 and S3, so the large initial C2 neural gain is not available to claim again. Before a CE6/epoch/architecture job, inspect residual disagreement by decoy type and report gain over a same-data no-new-CE control. A sibling versus no-sibling or raw-plus-normalized input comparison is evidence-motivated; blindly adding another highly correlated checkpoint is not. Freeze representation at inference unless retraining consistently (A16).

C2 routed about 18% of test targets and covered most false accepts/retrieved-owner misses. Our pipeline routes a different band of **pairs**; do not copy its target-level routing percentages or thresholds. Measure per-stage **oracle remaining truth, routed truth, useful CE captures, cost and new false accepts**. Distinguish “not retrieved”, “retrieved but never CE-scored”, “CE ranked wrong” and “correct pair rejected”. Wider routing is worthwhile only if that measured recoverable bucket clears the gate.

### R06 · high · Audit raw-score stacking and density-sensitive features together

**Reported evidence:** raw-fold-score stacking lost 0.0017/0.0044 in C2; fold models had different calibration and some context mixed models that had seen evaluated labels. Dropping graph summaries from its neural combiner gained +0.00019 (R:210,243–247). Those graph summaries are not necessarily the same as our target-group columns.

**Action:** keep A02's five-target-column ablation, now with stronger motivation, but add a fold- and country-stratified calibration check and a no-S3-context dense control (A04/A06). Compare deployed model averaging to training-time single-fold scores. Use properly cross-fitted calibration or rank/gap features only where their own label lineage is clean; ranks also change when competitors change. A normalized count or fraction may be more stable than a raw count, but its denominator changes too—test it rather than declare it density-proof.

Do not globally shift or quantile-match test CE logits to train. R:268–270 reports that routed-class composition explained an apparent neural score shift. `shiftfix.py` explicitly assumes stable class-conditional distributions; that assumption needs validation and is stronger than observing more low-confidence test pairs. A shifted marginal histogram alone cannot identify posterior miscalibration.

### R07 · medium · Separate retrieval opportunity from indistinguishable records

**Reported evidence:** C2's retrieval loss remained about 0.0041, but broad lexical rescues largely found shared-name, empty-address cases; 85% of one incremental set was in that difficult slice. Only 108 of 2,499 new owners were accepted by its old stage-1 model. Address-containment rescue had only a reported +0.00029 ceiling and needed roughly 69% capture with zero added false positives to clear a +0.0002 gate (R:253–260,282–291).

**Action:** re-rank rescue proposals by **incremental macro oracle ceiling × empirically observed scoring capture, then subtract false-accept/displacement cost**, estimating the full resulting macro score directly before promotion. The product is a planning heuristic, not an exact estimator because multiple recovered targets can interact within an anchor. Slice missing truth into unique/shared name, present/absent address, native script, number-role mismatch and current target conflict. A large pair-recall increase can have negligible macro benefit.

**Keep** the India dense result from REVIEW_1: its reported validation/public deltas are encouraging. Failed C2 name-only retrieval does not refute that different intervention. For US/France, measure dense unique truths and displaced lexical truths before increasing K. Use address containment/domain rescue only if this repo's own bounded oracle/capture check supports it; the retrospective's 172-owner count does not transfer.

“Irreducible” means no distinguishing evidence in the available fields/context under the tested setup, not a proof that every such case is impossible. Do not silently delete those records or stop scoring them. Stop expensive rescue work on a slice whose measured usable ceiling is below the action threshold.

### R08 · medium, conditional new source finding · Dense training needs ownership-aware positives

**Prompted by:** the retrospective's proposed multi-positive dense retrieval (R:308–311), combined with its observed one-owner/many-target structure.

**Source evidence:** `code/src/bienc_export.py:70–90` exports multiple true target pairs as only text columns `a,b`, losing owner IDs. `code/src/remote_bienc.py:50–56,70–79` shuffles pairs and uses diagonal labels for symmetric in-batch cross-entropy. **If two targets of the same owner enter one batch**, each is a true match for that anchor, but the off-diagonal true match is treated as a negative. Repeated anchor texts also conflict in the reverse direction. This is a conditional loss-design issue, not evidence that collisions are frequent or that the existing dense gain is invalid.

**Check before a costly retrain:** retain owner/target/source keys in export metadata, count same-owner batch collisions under the actual shuffle, and audit exact duplicate texts separately. Do not infer owner equality merely from names/text: distinct businesses can share them. IDs identify label groups/splits; they are not predictive input features.

**Later variant:** one-owner-per-batch sampling is a cheap control. A multi-positive objective can use `-log(sum(exp(s_ij), j in P_i) / sum(exp(s_ij), all eligible j))`, plus the reverse grouped-positive treatment, with owner-aware hard negatives. Specify whether the aim is at-least-one-positive retrieval or coverage of all true targets; compare mean-per-positive supervision if all-target coverage matters. Evaluate against the original loss at fixed pair budget, K, compute and downstream matching model. Unknown impact, low deadline priority unless collision counts are material. Avoid a late blind retrain of the winning embedder.

### R09 · medium · Run a controlled learning curve before another capacity sweep

**Reported evidence:** more stage-1 training data gave +0.00081 to +0.00119; typed-number features plus all data reached +0.00195 over the smaller baseline. Larger/smaller leaf counts did not help in those comparisons (R:203–206,231–232,264–265).

**Connection/action:** `code/src/stack.py:54,95–105` caps sampled pairs per partition at `MAX_PER_PART`; this is not C2's targets-per-fold knob. Report actual owners, positive/negative pairs and orphan diversity retained under each cap. Compare nested owner-aware samples at unchanged features, negative policy and validation, with sufficient boosting rounds and early stopping. Preserve all requisite true links and ownership boundaries. Refit dependent calibration/stack stages when the base-score distribution changes; reusing an old S3 is not a controlled deployment comparison.

Do this after the deadline if end-to-end regeneration cannot finish in time. Do not reinterpret a 100%-data refit with changed calibration as automatically better than the validated ensemble. Copying C2's 3M or 4000-round settings is not the lesson; measuring this pipeline's own learning curve is.

### R10 · medium · Transliteration and blocking are targeted options, not a replacement plan

**Reported evidence:** C2 used a training-only token dictionary covering a reported 86% of native-token occurrences, plus a fallback. Country/state blocking materially sped its TF-IDF route and improved recall, with country-wide fallback for unknown state (R:118–127). These are measurements for that retriever and normalizer.

**Action:** strengthen A14–A16's raw-text/domain/typed-number audit. For persistent native-script misses, compare a training-only, outer-fold-excluded transliteration view against the existing Unicode-name romanization and raw dense retrieval. Keep raw text available; no online transliteration/geocoding lookup is needed. Dictionary coverage alone does not establish matching recall.

Do not retrofit hard state blocking into this different budgeted lexical/dense pipeline just because it helped C2. First measure true-pair state conflicts, parser coverage, unknown-state fallbacks and null handling. Never drop unknown-country/state records silently. Country agreement and exclusive ownership should be rechecked against the exact supplied training files, then enforced as documented pipeline assumptions, not elevated into undocumented official output rules. Train-observed maxima of five/six targets per source are not safe hard prediction caps.

### R11 · medium · Simplify experimental decoding; preserve correctness fixes

**Reported evidence:** C2's segment/size thresholds, expected-F policy and listwise variants stayed within small changes; expected-F was approximately -0.00009 and size-aware thresholds +0.00028 (R:261–263). This supports down-ranking broad decoding research, not declaring every decoder equivalent.

**Changes:** A01 parity remains first; A18 deterministic ties remains a reproducibility requirement. A08's France sweep and A09's runner-up reassignment move behind measured density/score defects. Keep one simple per-target argmax-plus-calibrated-threshold control to show whether our current policy earns its complexity. Do not import C2's tau=0.70 to incompatible score populations.

The retrospective calls argmax+tau “exact”. Exclusive ownership gives a valid structural constraint, but **does not prove that independently thresholding targets globally maximizes macro F0.5**: targets assigned to the same anchor share a metric denominator. Likewise REVIEW_1's synthetic expected-F counterexample proves a mathematical approximation, not that replacing the production decoder raises the score. Prefer the validated simple control unless a measured, reproducible alternative wins.

### R12 · blocker for promotion · Treat artifact and resource discipline as part of the model

**Reported evidence:** C2 eventually used content-addressed normalization, fingerprinted stages, shard manifests, saved boosters, atomic/resumable outputs, deterministic ties and exact-output regeneration. Earlier file-existence checkpoint reuse and unsaved combiner models were real failures; full candidate validation exceeded laptop RAM (R:169–174,236–238,274–277).

**Action:** A10–A12 become an explicit promotion contract: exact source/dataset/split/feature/checkpoint/config fingerprints; ordered score-key digests; saved final boosters/calibrators; immutable prediction pair; official ID/subset validation; clean-directory inference regeneration; artifact SHA and environment/license manifest; and a recorded upload decision/result. Use chunked aggregation or a RAM-suitable machine for full artifacts. A smoke test or matching-only validation does not certify the full candidate file.

Keep one named owner per task and one designated uploader, with a small append-only decision log. Do not launch multiple overlapping runs against mutable cache names. The retrospective's agents, five-A100-hour runtime, external weight hosting and dependency versions are not an execution or licensing plan for this repo. Verify this package's actual model licenses, permitted weight distribution and accessible pinned artifacts; source comments and another team's zip PASS are insufficient.

## Revised task queue for Opus

All tasks below are **proposed/not run here**. Effort estimates assume caches and data are already available on the appropriate machine; they are not promises. Direct leaderboard gain is unknown for every item. Keep the best verified output immutable. Do not stack unvalidated changes merely because the deadline is near.

### Before the stated 21:00 freeze: five delivery work items, at most one experiment

- [ ] **N1 — correctness and freeze record, 10–15 minutes.** Run REVIEW_1 Appendix B on the exact winning dumps; compute affected-row counts and the same validation-policy comparison; record floor/tie policy and hashes. Start preserving required models and configs now. **Done when:** one named baseline, repeatable assembly, explicit parity decision, rollback artifact. Do not promote merely because a synthetic counterexample exists.
- [ ] **N2 — unified loss/density audit, 15–25 minutes.** Produce R01's country/slice report and a mutually exclusive or explicitly ordered macro-loss decomposition covering retrieval, CE routing, wrong winner, rejection and false acceptance. Include ownerless targets, unknown state and no-candidate anchors. **Done when:** the largest addressable loss bucket and meaningful graph shift are identified with counts; estimates are not mislabeled hidden truth. If that artifact already exists, reconcile rather than rerun.
- [ ] **N3 — at most one evidence-selected cached-score experiment, 15–30 minutes, conditional on the budget gate.** If graph/context drift dominates, consider the S3 target-feature ablation or dense no-S3-context control; if dense conflict dominates, consider gating dense additions against accepted lexical claimants. Choose only one, compare complete final lists, and inspect recovered versus displaced TPs. Prefer already-generated variants to any refit. **Done when:** keyed outputs and paired development deltas support promotion or rejection; limitations of existing label lineage remain explicit. Skip this item if N4 already supplies the selected candidate or the reserve is threatened.
- [ ] **N4 — triage dense/CE work, then make a bounded upload decision, 15–30 minutes after scores exist.** Compare already-produced US-only and France-only dense outputs with all other country lists fixed. CE6 enters if its artifacts already exist, a qualifying in-flight job fits, or it demonstrably earns the single prompt two-hour training slot under the directive above. Training/score lineage and the measured residual must justify it; a job finishing is not evidence of gain. Do not launch CE6 because this memo mentions it. A France abstention variant competes for the same single experiment slot as N3, not an extra slot. **Done when:** a decision-changing question, exact variant hashes and local evidence are logged before upload. Cancel speculative unfinished work that cannot complete validation before the time cutoff.
- [ ] **N5 — final reconstruction and validation, reserve 30–45 minutes plus full-run time.** Save final fitted stages; validate the exact matching/candidate files including IDs, duplicates and subset relationships; regenerate inference in a clean directory or state precisely what remains unverified; produce package manifest and upload-ready checksums. **Done when:** the artifact that will be handed in is the one that was tested. A package build alone does not prove final upload or full retraining reproducibility.

N1 and artifact preservation are mandatory correctness work; N2 informs the single choice across N3/N4 rather than those items becoming an automatic sweep. The budget directive above overrides every optional experiment in this document. If the corrected variant loses credible validation evidence, keep both artifacts and investigate; do not force a policy change solely to make implementations look alike. If time expires, prioritize N5 over unfinished exploratory work.

### Submission decision tree replacing REVIEW_1's nine-probe menu

Confirm the actual remaining allowance and interface precision. This is a selection guide for existing supported outputs and the one permitted experiment, **not permission to create one experiment for every bullet**. C2 reports a three-decimal display; our handoff reports six-decimal values. Do not assume either applies to the current upload path. A score display bucket is not a confidence interval.

1. Reuse existing equivalent results; first establish the frozen reference hash and affected country/row count.
2. Upload an A01 parity variant only if it changes enough rows to matter and labeled checks support it. Otherwise skip it as a leaderboard experiment while retaining the consistency fix for the next clean run.
3. If new dense outputs pass provenance/conflict checks, use only the unresolved US-only and/or France-only comparisons. If dense is unpromising, use **one** locally supported graph/calibration/France variant instead, not all three.
4. Submit a final supported combination only if it differs from an already scored candidate; validate interactions. Reserve at least three slots **if the stated allowance of twelve still holds**, including the final upload/corrections. Do not spend an allowance just because it exists.

The retrospective's blank-France probe is historical evidence about its older model under its own public weights/rounding assumptions. It does not reveal this pipeline's current France score, and it is not a recommendation to repeat a deliberately degraded submission. Use legitimate country-specific model improvements for delta evidence. No public probe reveals the private optimum, and a plateau within display rounding does not establish equivalence.

## After the deadline: dependency-ordered research roadmap

1. **A clean evaluation foundation.** Fresh outer holdout and full competitor scoring with excluded labels through every artifact; calibration/evaluation separation; representative natural orphans; deployed ensemble parity. Record both absolute performance and delta stability under real-data density perturbations. Existing consumed splits remain development data.
2. **Pair evidence plus learning curves.** Typed address roles and one-component twin evidence; source/script/empty-address tests; owner-aware increasing-data samples; adequate early stopping. Compare against current features and pair-only controls. A fresh fit must include downstream recalibration and score-model regeneration.
3. **Density-robust stacking.** Raw counts versus ratios versus no-graph controls, calibrated cross-fitted inputs, sibling/no-sibling contrast, complete lexical+dense competition. Verify benefits across density slices; do not optimize synthetic stress at the expense of original real data.
4. **Ownership-aware dense rescue.** Measure same-owner batch collisions; test multi-positive versus one-owner-per-batch objectives; hard negatives from real confusing owners/orphans; incremental oracle and scoring capture at fixed candidate budget. Preserve the current India win until a controlled replacement beats it.
5. **Complementary neural evidence and narrowly gated advanced work.** Test missing field evidence, a second epoch or complementary backbone only after disagreement/routing analysis. Measure address-containment capture before generating large candidate dumps. A frozen-embedding set-level head with NONE is an exploratory option, **must beat a same-input pair-only control**, and does not outrank the preceding evidence-backed work merely because code exists elsewhere.

For a modest planned experiment, a reasonable initial materiality gate borrowed from the retrospective is **+0.0002 raw macro-F0.5 on H_cal, with a positive paired uncertainty bound and no material regressions in critical slices**, subject to runtime and final-artifact feasibility. This is a proposed planning threshold, not a universal constant or statistical guarantee after many trials. Pre-register it, use the same anchors and artifact lineage, inspect practical error tradeoffs, and confirm selected variants on genuinely unconsumed evaluation data. Correctness, output validity and reproducibility fixes are not rejected simply because their measured gain is zero.

## Coverage and boundaries

This follow-up incorporates the retrospective's major themes: ownership/data-generator structure; natural decoys and density shift; France uncertainty; open and owner-grouped evaluation; derived-artifact leakage; raw-score stacking; typed number/sibling evidence; neural routing/complementarity; transliteration/state fallback; more-data learning curves; irreducible-looking ambiguity; negative retrieval/decoder/synthetic results; multi-positive and density-robust future work; compute constraints; artifact regeneration and submission discipline. Citations above tie each recommendation to the supplied text and, where relevant, this repo's implementation.

Do **not** carry across C2's fitted thresholds, weights, country scores, exact candidate counts, model-size totals, licensing conclusions or package status. Its reported ~57.5 candidates/anchor also does not update the ~21 figure for this repo, or settle the official interpretation of a multi-stage candidate boundary. Preserve REVIEW_1's explicit disclosure of the learned wide pool, c2, CE band and dense additions.

### Requested Opus response

In `opus/REPLY_1.md`, respond **first with the critical-path/running-job ledger and which explorations you stopped**, then N1–N5 and the current bottleneck using real artifacts. State which C2 lessons transfer, which fail for this architecture, actual remaining wall time/submissions, the winning source/dump hashes, and the chosen **single next experiment or explicit no-more-experiments decision** with its kill gate. Account for the user's four-hour wall-time budget and approximately two-hour H200 training cycle. For deferred items, distinguish no time, missing data, already tested and not applicable. Return measured deltas rather than adopting the retrospective's conclusions by analogy.

**Astra completed:** read and fingerprinted the entire supplied retrospective; reconciled its major results, negative evidence and units with REVIEW_1; inspected the relevant current source paths; refreshed upstream/PR status; revised the deadline queue and longer-term roadmap. No new real-data performance, package validation or retraining result is claimed.
