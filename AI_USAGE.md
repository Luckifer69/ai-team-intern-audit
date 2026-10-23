# AI usage

## How AI was used

**Claude Code (model: Claude Opus 5.5)** ran as an agent in my terminal from a detailed task prompt I
wrote. It did most of the work:
- read the PDFs and the starter kit;
- chose the corpus and tokenizers;
- wrote all scripts (`prepare_corpus.py`, `common.py`, `audit_fertility.py`, `token_cost.py`,
  `capacity.py`, `run_all.sh`);
- ran every experiment;
- drafted every document, including this one and NOTEBOOK.md.

All measured corpus statistics, tokenizer results and Part B calculations are traceable to saved
command output or reproducible arithmetic. Part C contains explicitly labelled assumptions,
estimates and forecasts; these are not claims of measured performance. Nothing was benchmarked on a
GPU; none was available.

That makes the parts I personally understand and can defend the real question here, not "did AI
help". The sections below are the AI's own error log, followed by my verification.

## Where the AI was wrong or misleading (caught during the session)

| # | what went wrong | how it was caught | fix |
|---|---|---|---|
| 1 | **Fabricated notebook timestamps.** Only the first NOTEBOOK heading time came from `date`. The AI *typed* plausible later times (21:10 … "00:05", implying 3 hours of work); the real span was ~31 minutes. | The AI checked `date` while writing checkpoint notes; file timestamps confirmed it. | First fix (2026-10-22): headings annotated with times reconstructed from file mtimes. An external review (2026-10-23) judged that insufficient. Now (Entry 20): headings are numbered entries, clock times only where read from `date`, Entries 2–15 say "time not recorded". The fabricated times and rough mtime windows are listed in Entry 20 and labelled as such. Entry order and content were real. |
| 2 | A summary table in AUDIT.md disagreed with E5: "hin −15%, mal +47%, sarvam 0.98×" vs E5's −18%, +41%, 0.95×; also "459 vs 99 tokens" vs 459/96. | Proofreading the summary against the JSON. | The AI "corrected" the summary to E5's numbers. **That fix was itself wrong** (see #10): E5 was confounded, and −15%/+47%/0.98× are the correctly isolated values. Now restored from the rewritten E5. The 459/96 fix stands. |
| 3 | Took the assignment's "parallel line-by-line" description of the samples on trust and labelled their 4.78× ratio "same content". **The samples are not parallel.** | Rehearsing a new-input run gave o200k hin/eng = 1.06 vs 1.57 on FLORES; the gap prompted a side-by-side check. | Finding D + experiment E10; label fixed in code and AUDIT.md. |
| 4 | Hypothesised NFC would be a no-op. It is not: it decomposes nukta letters and moves sarvam-1 Bengali by 14.6%. | Measured (E7). | Reported as a material sensitivity, not as "harmless". |
| 5 | Guessed lowercasing would *lower* English token counts. It raises them by 3.5%. | Measured (E2). | Direction stated from data. |
| 6 | Miscounted "e2e > wall clock in 11/13 rows" (it is 12/13). | Recount. | Correction entry in NOTEBOOK. |
| 7 | Initially assumed capping `max_num_seqs` always helps. A forecast built from measured rows says −2.8% for a 32-request burst. | Building the forecast from rows. | ANSWERS.md B2 states the cap does not help mild overload. |
| 8 | Wrote "13–32%" token savings; the correct range is 12–32%. | Recomputed from the summary. | Fixed. |
| 9 | Memo drafts were too long; the first one-page check used a renderer that flattened lists, so it was wrong too. | Rendered to PDF and inspected. | Fixed the renderer, then trimmed both memos to one A4 page. The renderer is now in the repo: `tools/render_memo.py`. |

## Where the AI was wrong (caught by an external review, 2026-10-23)

| # | what went wrong | how it was caught | fix |
|---|---|---|---|
| 10 | **E5 was not isolated.** It compared v0's full pipeline (split(" "), lowercase, NFC, per-sentence mean) with the corrected premium, yet AUDIT C1 called it "only the denominator changes". | External review. | E5 part 1 now uses the same token counts and changes only the denominator: distortion = W_eng/W_L − 1, identical across tokenizers (hin −14.6%, mal +46.7%; sarvam-1 hin 0.98 vs 1.14). The old comparison is kept as part 2, labelled "combined". |
| 11 | **Part C reviewer arithmetic was wrong.** It budgeted a hi/kn-only reviewer to QA SFT data in 6 languages. It also mixed A100 training numbers with L4 serving numbers and never stated that the L4 was assumed. SFT fallback scope was inconsistent. | External review. | Memo rewritten with a per-path reviewer ledger (580 / 880; 6-language QA needs 900, of which 600 can't be reviewed), A100 and L4 kept apart, L4 as stated assumption S1, SFT fallback limited to hi/kn. |
| 12 | B4 named two metrics (asked for one). The A3 cross-tokenizer index was worded like a cost. The A4 headline was stated more firmly than the evidence supports. | External review. | B4: one metric (`vllm:num_requests_running`). Index described as a token-count index. Headline softened; o200k called an interim proxy. |
| 13 | **The audit wrote `starter_kit/__pycache__/`** when it imported fertility.py via importlib. The hashed files were unchanged, but the starter dir was not left untouched. | Found while triaging the review (not one of its points). | Deleted. `sys.dont_write_bytecode = True` in the audit. `run_all.sh` step 6 now fails if the dir reappears. |

## Where the AI was wrong (caught by a second review, 2026-10-23)

| # | what went wrong | how it was caught | fix |
|---|---|---|---|
| 14 | E10 counted eng 7 / hin 10 as a "close" pair. English is an instruction ("Please keep the books in the cupboard"), Hindi a statement ("the books are kept in the cupboard"): related, not a translation. | Second review. | Moved to "partial" in `SAMPLE_PAIRS`; E10 rerun. Close-pair ratio is now gpt2 6.25 / o200k 1.39 / sarvam-1 1.04 (was 5.74 / 1.38 / 0.97, from 4 pairs). |
| 15 | Lowercasing was called an implementation bug, but `fertility.py` does it on purpose ("so casing doesn't add noise"). | Second review. | Reclassified as a measurement-design mismatch for serving-cost estimates. E2 numbers unchanged. |
| 16 | The A4 monitor (live tokens per request ÷ English) would move with traffic mix and reply length, not only the tokenizer. | Second review. | Replaced by the same-content premium on the deployed tokenizer over the fixed FLORES set (proposed; the deployed tokenizer wasn't available). |
| 17 | Part C said the rewriter was "dominated by" SFT; nothing measured either. Launch/iterate/kill rules were not stated together. | Second review. | (b) is "deferred: unvalidated", with three explicit outcomes. |
| 18 | Part C put the SFT fallback deadline on Day 15 (Thu 11-05), after the A100's last day (Day 14). | Found while making fix 17. | Moved to Day 14 (Wed 11-04). |
| 19 | This file claimed every number came from a command run. Part C numbers are assumptions and estimates. The audit docstring claimed every experiment changes exactly one factor. | Second review. | Both statements narrowed to what is true. |

One point from the first review did not match what was on disk: the reviewed ZIP contained caches and the raw tarball
at its root, but the zip built here on 2026-10-22 had neither. It was fixed anyway with
`make_zip.sh`, which builds *and* verifies the archive.

## Claims that rest on the AI's background knowledge, not verified in this session

- NVIDIA A100 bf16 dense peak ≈ 312 TFLOPS (Part C, labelled as an assumption).
- vLLM details: the metric names `vllm:num_requests_running` / `vllm:num_requests_waiting`; that
  preemption defaults to recompute; that the spec's `gpu_memory_utilization` / `max_model_len` imply a
  vLLM-like stack. All are labelled [A] in ANSWERS.md.
- Unicode details: rule GB9c, nukta letters being composition exclusions. The *effects* were
  measured; the explanation is background knowledge.
- **The Hindi glosses in AUDIT.md finding D are the AI's translations.** I must check them myself.

## What I verified myself

*(I fill this in honestly before submitting. Do not tick anything not actually done.)*

- [ ] Re-ran `./run_all.sh` on my machine and got the same outputs
- [ ] Re-derived B1 (114,688 B/token; 25 sequences) and B3 (200.9 tok/s, both ways) by hand
- [ ] Checked the Hindi glosses / sample alignment in AUDIT.md (finding D)
- [ ] Read `audit_fertility.py` E0–E10 and can explain why each before/after isolates one factor (E5 part 2 is deliberately the combined gap)
- [ ] Read both memos and agree with the recommendations (or changed them)
- [ ] Notes:
