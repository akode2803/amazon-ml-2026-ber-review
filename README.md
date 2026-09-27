# Amazon ML Challenge 2026 — Business Entity Resolution: code review repo

**Private. Do not make public** (active competition; the final ranking reviews code).

This repository exists so that **Astra** (GPT, reviewer) can critique our pipeline and **Opus** (Claude, the
engineer running it) can act on the critique. **Opus and Astra communicate through this repo only** — see
[`COMMS.md`](COMMS.md). Astra: start with [`ASTRA_HANDOFF.md`](ASTRA_HANDOFF.md).

| Path | What |
|---|---|
| `ASTRA_HANDOFF.md` | The brief for Astra: task, numbers, production path, what we want found. **Read first.** |
| `COMMS.md` | How Astra and Opus talk (files, naming, turn order). |
| `astra/` | Astra writes reviews here (`REVIEW_1.md`, `REVIEW_2.md`, …). |
| `opus/` | Opus writes replies / follow-up questions here (`REPLY_1.md`, …). |
| `code/src/` | All Python source (normalisation → blocking → stage 1/2 → cross-encoders → stage 3 → dense retrieval → decision). |
| `code/scripts/` | The exact shell drivers that produced every version (pod address redacted). |
| `docs/Amazon_ML_journey.md` | Full narrative: data, constraints, every experiment, leaderboard history. |
| `docs/problem_statement.pdf` | Official task statement and rules. |

No data, models, scores or outputs are committed (see `.gitignore`); every number quoted comes from our logs.
