# AI Team Intern Assignment — "The Audit": submission

Audit of `starter_kit/` (fertility.py, REPORT_v0.md, bench/). Start with the two memos
(`partA/memo.md`, `partC/memo.md`) and `partB/ANSWERS.md`. Everything else is supporting
evidence.

## Layout

```text
ai-team-intern-audit/
  README.md               this file: setup, reproduction, rubric matrix
  NOTEBOOK.md             numbered lab notebook (read the timestamp disclosure at its top)
  AI_USAGE.md             what the AI did, where it was wrong, what to verify
  CHECKPOINTS.md          checkpoint status notes + working rubric matrix
  DEFENSE.md              live-defense sheet: equations, Q&A, counterfactuals, risks
  STARTER_KIT_SHA256.txt  hashes proving starter_kit/ is unmodified
  requirements.txt        pinned deps (Part B needs only the stdlib)
  run_all.sh              clean reproduction of every number
  make_zip.sh             builds ../your-submission.zip and verifies its contents
  tools/render_memo.py    one-page check for the memos (md → PDF via LibreOffice)
  partA/
    memo.md               A4 one-page recommendation
    CORPUS.md             A1 corpus: source, license, stats, preprocessing, limits
    AUDIT.md              A2 findings, each with experiment + before/after
    ANALYSIS.md           A3 corrected comparison + the single routing metric
    scripts/              prepare_corpus.py, common.py, token_cost.py
    audit/                audit_fertility.py (E0–E10) + saved outputs
    results/              summary.csv, per_sentence.csv, run_meta.json, stdout
    data/flores200_devtest/  8 FLORES-200 devtest files (CC-BY-SA 4.0, NOTICE) + manifest.json
  partB/
    ANSWERS.md            B1–B4 written answers
    capacity.py           stdlib-only calculations (row selectors, what-if flags)
    capacity_output.txt, results.json
  partC/
    memo.md               one-page decision memo
```

## Reproduce

Requires bash, `uv`, `zip`, and internet access on the first run (FLORES tarball 25.6 MB plus
tokenizer files). The starter kit is **not** in this repository. The scripts expect it as a sibling folder named
`starter_kit/`:

```text
workspace/
├── ai-team-intern-audit/ (or your-submission/)
│   ├── README.md
│   ├── run_all.sh
│   └── ...
└── starter_kit/          (the assignment's starter-kit ZIP)
    ├── fertility.py
    ├── REPORT_v0.md
    ├── bench/
    └── corpus_sample/
```

```bash
mkdir workspace && cd workspace
unzip /path/to/your-submission-final.zip    # creates your-submission/
unzip "/path/to/starter_kit.zip"            # creates starter_kit/ (and a __MACOSX/ folder to ignore)
cd your-submission
uv venv --python 3.12 .venv                 # run_all.sh does these two steps itself if .venv is missing
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate                   # optional: the scripts call .venv/bin/python directly
./run_all.sh        # checks starter hashes, rebuilds corpus, runs baseline, audit, A3 and Part B (~35–90 s)
./make_zip.sh your-submission-final.zip     # writes ../your-submission-final.zip and verifies it
```

Windows: run these in WSL or another bash environment; native PowerShell is not tested.

Verified 2026-10-23 after the first review's fixes, from a clean state (venv, caches, raw tarball and
every generated output deleted): exit 0 in 37 s, and only E5 changed. After the second review's fixes,
these exact steps were rerun in an empty workspace, from the final ZIP and the original starter-kit ZIP:
exit 0. Outputs were identical to the shipped ones apart from absolute paths in log lines; only E10 changed. The run also
re-checks the starter hashes at the end and fails if `starter_kit/__pycache__` appears.
Individual commands are in `DEFENSE.md` §5.

Package: `./make_zip.sh [NAME.zip]` writes `../NAME.zip` (default `your-submission.zip`) with a
top-level `your-submission/`. It excludes the venv, tokenizer caches, raw tarball and bytecode, then verifies the archive: CRC, no
excluded paths, and every shipped file present.

One-page check: `.venv/bin/python -I tools/render_memo.py partA/memo.md partC/memo.md` (needs
LibreOffice; renders A4, 11 pt, 2 cm margins, prints the page count).

**Environment actually used:**
- Arch Linux, CPython 3.12.15 via uv, tiktoken 0.14.0, tokenizers 0.23.3, transformers 5.20.2
  (no torch), regex 2026.9.29.
- **No CUDA GPU** (AMD integrated graphics only). Nothing was benchmarked; Part B is arithmetic over
  the supplied CSV.

**Evidence labels used in the documents:**
- **[S]** source fact
- **measured** — output of a command here
- **[C]** calculation
- **[A]** assumption
- **[H]** hypothesis
- **[F]** forecast

## Rubric matrix (self-assessed, after the 2026-10-23 review fixes)

These are my own statuses. **\*** = fixed after the external review and not yet re-reviewed by anyone else.

| item | status | evidence | reason / caveat |
|---|---|---|---|
| A1 corpus (10) | **PASS** | `partA/CORPUS.md`, `data/flores200_devtest/manifest.json`, `scripts/prepare_corpus.py` | Real FLORES-200 devtest: 8 langs × 1,012 aligned sentences (eng, hin, kan, tam, tel, mal, ben, mar). License verified upstream; sha256 for tarball and files; sanity checks; limits paragraph. |
| A2 audit (20) | **PASS\*** | `partA/AUDIT.md`, `audit/audit_fertility.py`, `audit/audit_output.txt` | E0 proves harness fidelity. **C1 (E5) is now isolated:** same token counts, only the denominator changes. Distortion W_eng/W_L: hin −14.6%, mal +46.7%; sarvam-1 Hindi sign flip 0.98 vs 1.14. The combined v0→corrected gap is labelled as such. Also C2 code-point "char"; bugs (split(" ") ≤2.2%, 2 crashes); lowercasing (+3.5% English) is a measurement-design mismatch, not a bug; harmless (random.seed, blank-line skip); NFC valid but material. **Open:** finding D's Hindi glosses are AI-written, so the candidate must check them. |
| A3 analysis (12) | **PASS\*** | `partA/ANALYSIS.md`, `results/summary.csv` | 3 tokenizers (gpt2, o200k_base, Indic sarvam-1); 4 denominators; pooled + sentence-level + bootstrap CI + 2 sensitivities. Single metric = same-content token premium on the production tokenizer (o200k as interim proxy). The cross-tokenizer index is a token-count index, not a cost. |
| A4 memo (8) | **PASS\*** | `partA/memo.md` | 1 A4 page (`tools/render_memo.py`). Headline no longer treats 6× as established. Routing recommendation, caveat, one monitoring metric: same-content token premium on the deployed tokenizer over the fixed FLORES set (proposed; not measurable here). |
| B1 (7) | **PASS** | `partB/ANSWERS.md`, `capacity.py` | 114,688 B/token; 25 sequences. Matches log: implied 105.7k tokens; preempted = batch − 25. |
| B2 (6) | **PASS** | same | Rows 12–14 cited; mechanism has 3 supporting observations. `max_num_seqs=24` effect is a labelled **forecast** (no GPU). |
| B3 (4) | **PASS** | same | Counter includes prompt tokens (13/13 rows). Goodput 200.92 / 200.93 tok/s by two independent routes. |
| B4 (3) | **PASS\*** | same | One metric: `vllm:num_requests_running`, expected ≈25 in decode (≤29 near prompt length), never 32/48. Metric name from background knowledge, labelled [A]. |
| C memo (15) | **PASS\*** | `partC/memo.md` | 1 page. Paid hi/kn reviewer ledger (580 prompt path / 880 with SFT fallback; 6-language QA infeasible). A100 training and L4 serving (stated assumption S1) kept separate. SFT fallback hi/kn only; rewriter deferred (unvalidated), not called inferior. Numeric success metric; launch / iterate / kill rules with kill criterion Thu 2026-10-29; day-one experiment. All figures are assumptions/proposals; nothing was run. |
| NOTEBOOK | **PARTIAL** | `NOTEBOOK.md` | Content, order, dead ends and corrections are genuine. Headings 2–15 originally had **AI-invented clock times**. They are now numbered entries with "time not recorded"; clock times appear only where read from `date`; the invented times are disclosed in Entry 20. |
| AI_USAGE | **PARTIAL** | `AI_USAGE.md` | AI error log complete, including errors caught by two external reviews (#10–19). **The candidate's own verification checklist is still blank and must be filled in by the candidate.** |
| Defense (15) | **PARTIAL** | `DEFENSE.md` | Sheet, rerun recipes and tested flags are ready (updated for E5, B4, Part C). Readiness depends on the candidate working through it. |

## Main limitations

- **Tokens:** the FLORES domain (formal, translated) and the prompt-only measurement limit what the
  token numbers say about production. A live metric to catch this is proposed in the A4 memo.
- **Part B:** no confirmatory benchmark could be run. Two CSV columns (`e2e_ms_p95`, `ttft_ms_p50`)
  look physically inconsistent and are deliberately not used.
- **Part C:** quantities are assumptions; the reviewer rate and model access are the most
  decision-sensitive.
