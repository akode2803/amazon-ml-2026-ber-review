# COMMS — how Opus (Claude) and Astra (GPT) talk

The repo **is** the channel. No other channel exists between us; everything must be in a committed file.

## Turn order
1. **Astra** reads `ASTRA_HANDOFF.md` + the code, writes **`astra/REVIEW_1.md`**, commits and pushes.
2. **Opus** pulls, verifies each finding against the running system, writes **`opus/REPLY_1.md`**
   (what was confirmed / refuted, measured effect, follow-up questions), pushes.
3. **Astra** answers in **`astra/REVIEW_2.md`** — only if it has something new. Then round 3 if time allows.

## Rules
- One file per turn, never edit the other side's files. Append-only.
- Put the **IST time** at the top of every file (deadline 23:00 IST, 27 Sep 2026; code freeze ~21:00).
- Reference code as `code/src/<file>.py:<line>` (line numbers of the commit you read).
- Findings format (per finding): **ID · severity (blocker/high/medium/low) · expected leaderboard impact
  (points of F0.5, with your confidence) · effort (minutes) · evidence (file:line + reasoning) · concrete fix.**
- Rank findings by *expected leaderboard gain per minute of work*. Opus has ~4 hours of wall-clock left.
- Astra's usage is limited (~2–3 hours total): **front-load the highest-value findings in REVIEW_1**; do not
  hold anything back for a later round.

## Commit hygiene
Commit messages: `astra: REVIEW_1 …` / `opus: REPLY_1 …`. Never commit data, models or outputs.
