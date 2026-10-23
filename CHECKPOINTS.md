# Checkpoints

Status notes per checkpoint (append-only), all from 2026-10-22. Detailed evidence lives in NOTEBOOK.md.
**Current status is the rubric matrix in `README.md`.** This file is history: it records what the state
was at each checkpoint, and later review fixes (NOTEBOOK Entries 19+) are not reflected here.
**Disclosure:** these headings originally carried clock times that the AI made up, not read from `date`, and they were
wrong (NOTEBOOK Entry 16). No exact times were recorded, so each checkpoint is now tied to the NOTEBOOK entry
it follows. All six were reached on 2026-10-22 between Entry 1 (21:04, `date`) and Entry 16 (21:36, `date`).
CP1–CP3 notes were written together after Entry 8; CP4–CP6 together after Entry 15.

## Rubric → evidence matrix: historical snapshot at CP3 (not the current status; see README.md)

| item | requirement | evidence / output path | acceptance | status @CP3 |
|---|---|---|---|---|
| A1 (10) | ≥4 langs incl. eng, hin, 2 Dravidian; source, license, size, domain, preprocessing, limits | `partA/scripts/prepare_corpus.py`, `partA/data/flores200_devtest/manifest.json`, `partA/CORPUS.md` | real sourced text, sha256, sanity checks, limits paragraph | data built; CORPUS.md pending |
| A2 (20) | bugs + ≥1 conceptual error + ≥1 harmless thing, each with isolated before/after | `partA/audit/audit_fertility.py`, `audit_output.txt`, `audit_results.json`, `partA/AUDIT.md` | E0 fidelity proof; every claim has command + numbers | experiments done; write-up pending |
| A3 (12) | ≥2 tokenizers (one Indic-aware), ≥2 denominators, single routing metric + why | `partA/scripts/token_cost.py`, `partA/results/*.csv` | reproducible tables, pooled + sentence-level, CI | pending |
| A4 (8) | ≤1 page memo: numbers, routing rec, caveat, 1 production metric | `partA/memo.md` | ≤1 page, all 4 elements | pending |
| B1 (7) | KV bytes/token exact; max 4096-seq concurrency; check vs log | `partB/capacity.py`, `partB/ANSWERS.md` | formula + units; cross-check rows | arithmetic done in notebook; script pending |
| B2 (6) | anomaly rows/cols, mechanism, 1 change + forecast | same | cites rows; forecast labelled | pending |
| B3 (4) | misread column, goodput 2 ways, what report should say | same | two derivations agree | verified inline (200.9 tok/s); script pending |
| B4 (3) | one counter + expected value | same | discriminates hypotheses | pending |
| C (15) | ≤1 page; assumptions, arithmetic, numeric success metric, dated kill criterion, day-1 experiment | `partC/memo.md` | all labelled parts present | pending |
| Defense (15) | re-derive, rerun, modify, counterfactuals | `DEFENSE.md`, README commands | scripts take new paths/flags | pending |

## CP1 — after NOTEBOOK Entry 3 — inventory + baseline

- Done: all starter files read; PDFs read (page 5 rendered to confirm clipped tree is in the source);
  originals hashed (`STARTER_KIT_SHA256.txt`); env recorded (no CUDA GPU; 7 GB RAM; network OK);
  baseline `fertility.py` reproduces REPORT_v0 table exactly; `reported_tok_s` semantics verified on 13/13 rows.
- Blockers: none. No GPU ⇒ Part B is arithmetic-only (as intended by the assignment).
- Next: corpus source + license verification.

## CP2 — after Entry 6 — plan + corpus licensing verified

- Matrix above. Corpus: FLORES-200 devtest, CC-BY-SA 4.0 verified from upstream README.
- Tokenizers: gpt2, o200k_base (tiktoken), sarvamai/sarvam-1 @e9607337 (research license — OK for
  measurement, flagged for production).
- Next: build corpus, audit harness.

## CP3 — after Entry 8 — A1 data + A2 evidence exist

- Done: corpus built + checks (manifest.json); audit E0–E9 run, outputs saved.
- Uncompleted: CORPUS.md, AUDIT.md write-ups; A3; Part B script; memos; C; defense sheet.
- Next highest-value action: A3 corrected analysis (feeds A4 memo).

## CP4 — after Entry 10 — A3 + Part B validated

- A3: `partA/results/` (3 tokenizers × 8 langs; pooled + sentence-level + bootstrap CI; NFC and
  whitespace sensitivities); sarvam-1 loader ≡ transformers on 8,096 sentences.
- Part B: `partB/capacity.py` + independent awk recomputation; B3 two ways agree (200.92 vs 200.93).
- Next: write-ups (CORPUS/AUDIT/ANALYSIS/ANSWERS), memos.

## CP5 — after Entry 13 — memos done

- partA/memo.md and partC/memo.md each render to 1 A4 page (11 pt, 2 cm). Part C has assumptions,
  arithmetic, decision/trade-offs, numeric success metric, dated kill criterion (Thu 2026-10-29),
  day-one experiment. Date anchor = machine date 2026-10-22 (stated in memo).
- Late finding (Entry 14): starter samples not parallel → finding D / E10; corrected an earlier
  mislabel. Nothing else depended on it.

## CP6 — after Entry 15 — clean validation

- `./run_all.sh` from a clean state: exit 0, all outputs byte-identical, starter kit hashes OK.
- Remaining: README (final rubric matrix), AI_USAGE.md, DEFENSE.md, zip + archive check.
