# A2 — Audit of `fertility.py` and REPORT_v0 §1

All experiments are in `partA/audit/audit_fertility.py`. Run them all with:

```bash
.venv/bin/python -I partA/audit/audit_fertility.py            # ~3 s; or: --only E1 E7
```

Saved output: `audit/audit_output.txt`, `audit/audit_results.json`. Corpora: the two starter
samples, plus the A1 FLORES devtest files (1,012 sentences per language). Tokenizers: gpt2 (the v0
setting), o200k_base, sarvam-1 (see A3).

**Method.** `analyze_variant()` re-implements `read_lines` + `analyze` with one switch per suspected
issue.
- **E0** shows that with every switch at the v0 default it reproduces the original
  `fertility.analyze()` **bit-for-bit** on all 10 corpora (e.g. sample hin = 7.448452 / 1.579108).
- E1–E4 and E7 each flip exactly one switch, so each of those deltas is caused by that switch alone.
  E5 part 1 holds the token counts fixed and changes only the denominator. E5 part 2 is labelled as
  a combined comparison. E6, E8, E9 and E10 are direct tests (diff, injected lines, crash inputs, line
  matching).

## Summary

| # | claim | class | effect on v0 numbers | changes a ranking/decision? |
|---|---|---|---|---|
| C1 | Per-word fertility ratio is used as a per-request cost ratio, but words are not the same amount of content across languages | **conceptual / metric error** | isolated (same tokens): hin −14.6%, mal +46.7% for every tokenizer; sarvam-1 hin 0.98× vs 1.14× (sign flip) | **yes** |
| C2 | "tok/char" counts Unicode code points; the "two metrics agree" robustness argument is invalid | **conceptual / metric error** | same samples: hin/eng = 7.00× (code points), 10.86× (graphemes), 2.65× (bytes) | **yes** (the report's corroboration argument) |
| R | REPORT claim "property of the script, not the tokenizer" | **interpretation error** (report, not code) | same Hindi text: 7.41× / 1.57× / 1.14× by tokenizer | **yes** |
| D | the "parallel" starter samples are not line-aligned (only 3 of 10 Hindi lines have a close counterpart, all at other line numbers; 2 more are only partial) | **input-data error** | hin/eng token ratio on the 3 close pairs vs all lines: gpt2 6.25 vs 4.78, o200k 1.39 vs 1.06 | yes, for any per-request claim from the samples |
| B1 | `line.split(" ")` counts an empty "word" per extra space | implementation bug | fertility deflated by up to 2.0% (gpt2: kan, hin sample), 2.2% (sarvam-1 kan) | no |
| B2 | `.lower()` before encoding (intentional in v0) counts tokens of text that is never served | **measurement-design mismatch** for serving-cost estimates | English tokens +3.5% (gpt2) ⇒ every Indic/English ratio understated ~1.5–3.5% | no |
| B3 | crashes: `<|endoftext|>` in text; blank-only file | robustness bug | none on reported numbers | no |
| N1 | mean of per-line ratios instead of ratio of totals | valid choice, minor | ≤1.5% (FLORES), ≤2.1% (samples) | no |
| N2 | NFC normalisation | **valid, but not neutral** | ≤1% for gpt2/o200k; **sarvam-1 Bengali 14.6%** | only for sarvam-1 Bengali |
| H1 | `import random` / `random.seed(1337)` | **harmless** (dead code) | output byte-identical without it | no |
| H2 | `strip()` and skipping blank lines | **harmless** for v0 | output identical after injecting 14 blank lines | no |

---

## C1 — Per-word fertility answers the wrong question (conceptual)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E5`

**Claim.** The code computes tokens per whitespace word correctly. REPORT_v0 then treats the
Hindi/English fertility ratio as "cost per request". A request is a fixed amount of *meaning*, not a
fixed number of words. Languages express the same sentence in very different numbers of words: on
FLORES, words per sentence are eng 21.6, hin 25.3, kan 15.9, mal 14.8. That is because Hindi uses
separate postpositions, while the Dravidian languages are agglutinative.

**Experiment (E5, part 1 — isolated).** For each tokenizer and language I hold one preprocessing
configuration fixed: corrected text handling (`split()`, case kept, no NFC, pooled sums). From the
**same token counts** I compute two numbers:
- the per-word ratio `(T_L/W_L) / (T_eng/W_eng)`;
- the same-content premium `T_L / T_eng`, i.e. tokens for the same parallel sentences.

Algebraically, per-word ratio ÷ premium = `W_eng / W_L`, so **only the denominator differs**. The
distortion is set purely by words per sentence and is identical for every tokenizer.

| | hin | kan | tam | tel | mal | ben | mar |
|---|---|---|---|---|---|---|---|
| words/sentence (eng = 21.6) | 25.3 | 15.9 | 16.6 | 16.7 | 14.8 | 19.3 | 18.8 |
| gpt2 per-word ratio | 6.33 | 18.48 | 20.28 | 16.77 | 22.24 | 10.73 | 9.03 |
| gpt2 same-content premium | **7.41** | **13.59** | **15.54** | **12.97** | **15.16** | **9.56** | **7.86** |
| sarvam-1 per-word ratio | **0.98** | 1.65 | 1.51 | 1.49 | 1.99 | 1.44 | 1.23 |
| sarvam-1 same-content premium | **1.14** | 1.22 | 1.16 | 1.15 | 1.36 | 1.28 | 1.07 |
| **distortion (all 3 tokenizers)** | **−14.6%** | **+36.0%** | **+30.6%** | **+29.3%** | **+46.7%** | **+12.3%** | **+15.0%** |

Robustness: rerunning part 1 with v0's text handling (`split(" ")`, lowercased, NFC, still pooled)
gives distortions of −14.6% / +32.9% / +29.8% / +27.5% / +46.0% / +11.8% / +14.5% (same order), so
the conclusion doesn't depend on preprocessing. o200k: per-word 1.34 vs premium 1.57 for Hindi,
2.87 vs 1.96 for Malayalam (`audit_output.txt`).

**E5, part 2 — combined correction (NOT an isolated experiment).** This compares what the full v0
pipeline prints with what the full corrected pipeline reports. Four settings change *and* the
metric changes, so it shows the total gap, not what causes it:
- gpt2: hin 6.11 → 7.41, mal 21.43 → 15.16;
- sarvam-1: hin 0.95 → 1.14.

The starter samples are excluded because they are **not parallel** (finding D below); their
total-token ratio (4.78×, gpt2) compares different sentences.

**Direction and magnitude** (part 1):
- Per-word ratios **understate** Hindi by 14.6%, because Hindi uses more words per sentence than
  English.
- They **overstate** the Dravidian languages by +29% to +47%, and Bengali/Marathi by +12% to +15%,
  because those languages use fewer words per sentence.
- The ranking changes. gpt2 per-word ranks mal (22.24) > tam (20.28); same-content premium ranks
  tam (15.54) > mal (15.16).
- **Sign flip on sarvam-1:** per word, Hindi looks *cheaper* than English (0.98×), but the same
  sentences cost 14.4% *more* (1.14×).

**Why the comparison proves it.** Numerator tokens are identical. The two ratios differ by exactly
`W_eng/W_L`, in both directions, and by the same percentage for all three tokenizers. So the
per-word ratio mixes tokenizer efficiency with each language's morphology.

## C2 — "tok/char" is code points; agreement between the metrics proves nothing (conceptual)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E4`

**Claim.** `chars = len(line)` counts Unicode code points. In Indic scripts one written character
(an akshara) is several code points: consonant + vowel sign, plus virama for conjuncts. A code point
is an encoding artefact, not a unit of content. REPORT_v0 says the tok/char ratio (7.0×) "confirms"
the per-word ratio (5.89×), so "the result is robust". But both are language-internal denominators.
The agreement is coincidental, and the per-char number depends on how "character" is defined.

**Experiment (E4).** Same samples, same gpt2 tokens. Only the definition of "char" changes:

| unit | hin/eng ratio of tok-per-unit |
|---|---|
| code points (v0) | **7.00×** |
| extended grapheme clusters | **10.86×** |
| UTF-8 bytes | **2.65×** |

Minimal example: "किताबें" is 7 code points, 3 graphemes and 21 bytes; "books" is 5/5/5.

**Direction and magnitude.** The same token counts support any ratio from 2.65× to 10.86×.
Switching to graphemes raises Indic tok/char by 53–89%; switching to bytes lowers it by about 62%
(FLORES, all tokenizers). The "per character" ratio carries no cost information on its own.
Graphemes are not script-neutral either. `regex` `\X` keeps Devanagari conjuncts as one grapheme
(`क्ष` → 1) but splits Kannada ones (`ಕನ್ನಡ` → 4) under Unicode 15.1 rule GB9c.

## R — "A property of the script, not the tokenizer" (report interpretation)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E5`

Tested in E5/A3 on the same 1,012 Hindi sentences. Same-content ratio vs English:
**7.41× (gpt2), 1.57× (o200k_base), 1.14× (sarvam-1)**. The script is constant across these three
runs and the ratio moves 6.5×, so the tokenizer is the dominant factor. The Indic-aware tokenizer
nearly closes the gap. A related report claim, "budget 6× for Hindi", rests on gpt2, which is not
the serving model's tokenizer.

## D — The "parallel" sample corpora are not parallel (input data)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E10`

The assignment describes `corpus_sample/` as "English + Hindi, parallel line-by-line". It isn't.
Side by side (`paste`, NOTEBOOK Entry 14): **at the same line number**, only line 3 corresponds,
and only partially. Three Hindi lines have a close counterpart at a *different* line number, and
one more (hin 10) is related to eng 7 but is not a translation of it. My glosses of the Hindi lines
(please verify; I am the translator here) are below. Pair numbers are 1-based line numbers,
(eng, hin), as in `SAMPLE_PAIRS` in `audit_fertility.py`.

| hin line | gloss | English counterpart |
|---|---|---|
| 1 | I like morning tea very much. | — (eng 1 is about an airport) |
| 2 | It is raining lightly in Bengaluru today. | — |
| 3 | I bought this book just yesterday. | eng 3, *partial*: English adds "from a small shop near MG Road" |
| 4 | We are going to Mysore next week. | eng 8 |
| 5 | Have you eaten? | — |
| 6 | The train arrived on time. | eng 5 |
| 7 | Children are playing cricket in the field. | eng 4 |
| 8 | I need some water. | — |
| 9 | There was a lot of work at the office today. | — |
| 10 | The books are kept in the cupboard. | eng 7, *partial*: English is an instruction ("Please keep the books in the cupboard."), Hindi a statement |

The five unpaired English lines are heavy with named entities and acronyms (Bengaluru International
Airport, Quarterly Review, NASA, ISRO, GPU). The unpaired Hindi lines are short everyday sentences.

**Experiment (E10).** Total Hindi tokens ÷ total English tokens: all 10 lines by line number vs
content-matched pairs (text as written). Close pairs (eng, hin): (8, 4), (5, 6), (4, 7). Partial:
(3, 3), (7, 10).

| tokenizer | all 10 lines | 3 close pairs | 3 close + 2 partial |
|---|---|---|---|
| gpt2 | 4.78 | **6.25** | 5.15 |
| o200k | 1.06 | **1.39** | 1.18 |
| sarvam-1 | 0.79 | **1.04** | 0.83 |

**Direction:** the mismatch makes Hindi look cheaper than on matched content:
- 23.5–24.4% cheaper than on the 3 close pairs (gpt2 −23.5%, o200k −23.5%, sarvam-1 −24.4%);
- 5–10% cheaper if the 2 partial pairs are included.

The cause is that the unmatched English lines are token-heavy. **Caveat:** with 3 close pairs (24–26
English tokens) this only illustrates the bias; no rate should be estimated from it. A few related
pairs do not make the samples parallel. This is why A1 builds a real parallel corpus. v0's
per-language averages are not themselves affected by alignment (see H2), but REPORT_v0's
"6× more per request" is a per-request claim, and these files cannot support one.

## B1 — `split(" ")` counts empty words (implementation bug)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E1`

- **Minimal (E1):** `"please keep the books  in the cupboard."` → `split(" ")` gives 8 items
  including `''`; `split()` gives 7.
- **Before/after:** only the word denominator changes; fertility goes from v0 to fixed.
  - samples: eng 1.2652 → 1.2831 (+1.41%), hin 7.4485 → 7.5985 (+2.01%)
  - FLORES gpt2: kan +1.99% (213 lines with double spaces), tel +1.28%, tam +0.50%, eng 0.00%
  - o200k and sarvam-1: same pattern, up to +2.2% (kan)
- **Direction:** v0 *underestimates* fertility wherever double spaces occur. `split(" ")` would also
  fail to split on tabs or NBSP; that doesn't occur in these corpora (0 lines with non-space
  whitespace).
- **Magnitude:** ≤ 2.2%. No ranking changes.

## B2 — Lowercasing before tokenization (measurement-design mismatch, not a coding error)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E2`

- **Minimal (E2):** "NASA and ISRO announced a joint mission update." is 10 tokens as written and 11
  lowercased, under all three tokenizers.
- **Before/after:** tokens of the text as served, instead of lowercased text. English FLORES with
  gpt2: 27,994 (v0, lowercased) → 27,044 (as written), so lowercasing inflates English by **+3.5%**.
  o200k +1.6%, sarvam-1 +1.6%.
  - Indic changes are < 0.1%. The scripts are caseless; only embedded Latin named entities are
    affected (Telugu changes most, −0.03% to −0.10%).
- **Direction:** v0 overstates English tokens, so it understates every Indic/English ratio by about
  3% on gpt2. I had expected the opposite direction before measuring.
- **Magnitude:** ≤ 3.5%. No ranking changes: E2's per-language token totals give the same order
  with and without lowercasing on all three tokenizers. Every Indic/English ratio rises by the same
  1.5–3.5%.
- **Classification.** The code does what it intends: `fertility.py` line 59 says "lowercase so casing
  doesn't add noise to the comparison". That is a defensible choice for comparing tokenizers. It is
  the wrong choice for **estimating serving cost**, because production requests are tokenized with
  their original casing. The discrepancy matters only when the numbers are read as serving cost, as
  REPORT_v0 does, and only for text with cased (Latin) characters: English and embedded named
  entities. It is not an implementation defect.

## B3 — Crashes (robustness bug; E9)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E9`

- Text containing `<|endoftext|>` raises `ValueError: Encountered text corresponding to disallowed
  special token`. `tiktoken.encode` disallows special tokens by default; `encode_ordinary` is the fix.
- A file with only blank lines raises `ZeroDivisionError`.

Both return code 1. Neither affects the numbers in REPORT_v0. `scripts/common.py` uses
`encode_ordinary`, and `token_cost.py` exits with a message on an empty sentence.

## N1 — Mean of per-line ratios (valid, minor)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E3`

E3 replaces `mean(tokens_i / words_i)` with `sum(tokens) / sum(words)`. The change is at most 1.5% on
FLORES and at most 2.1% on the samples. The v0 line-mean is consistently a little higher because
short lines have higher fertility. Cost adds up tokens, so the pooled ratio is the cost-relevant
aggregate, and A3 uses it. The difference is too small to call the v0 choice a bug.

## N2 — NFC normalisation: looks harmless, is not neutral (valid but material)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E7`

The comment says "normalize just in case". I expected a no-op and was wrong. FLORES Bengali has 609
non-NFC lines; Hindi has 93. NFC **decomposes** precomposed nukta letters such as য় U+09DF → U+09AF
U+09BC, because they are Unicode composition exclusions.

- **Minimal (E7):** U+09DF alone is 4 tokens under sarvam-1 (byte fallback) and 2 after NFC.
  Under gpt2 it goes 3 → 4.
- **Before/after:** turning NFC off gives Bengali tokens of 258,460 vs 259,783 with NFC on gpt2
  (NFC +0.5%), and o200k NFC +0.9%. On **sarvam-1** it is **40,282 raw vs 35,149 with NFC**: raw text
  costs +14.6%, and Hindi +1.1%.

**Verdict.** NFC is canonically equivalent text, so applying it is legitimate. But it is a
measurement condition. The numbers are valid only if production applies the same normalisation
before tokenizing. A3 reports raw text as primary and NFC as a sensitivity column.

## H1 — `random.seed(1337)` (harmless)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E6`

- Only two lines mention `random`: the import and the seed (E6).
- Deleting both gives byte-identical stdout, and the original is deterministic across runs.
- Nothing in the pipeline samples, so the seed is dead code. It is not a bug and does not need
  fixing.

## H2 — `strip()` and skipping blank lines (harmless for v0)

**Command:** `.venv/bin/python -I partA/audit/audit_fertility.py --only E8`

E8 injects 14 blank and whitespace-only lines into each sample, and v0 output is identical. v0
averages within each language independently and never pairs line *i* across files, so dropping blank
lines cannot misalign anything. The same behaviour *would* break a per-sentence parallel comparison,
which is why `common.read_corpus` keeps blank lines and `token_cost.py` refuses empty sentences.

---

### What I checked and do **not** claim

- The "5.89×" ratio is computed from unrounded values (7.448/1.265). It is not an arithmetic slip
  (NOTEBOOK entry "Baseline reproduction").
- The order dependence of the ratio (the first `--corpus` is the base) is intended behaviour.
