# A3 — Corrected multilingual comparison

```bash
.venv/bin/python -I partA/scripts/token_cost.py          # ~12 s; writes partA/results/
```

Outputs: `results/summary.csv` (one row per tokenizer × language), `results/per_sentence.csv`
(24,288 rows), `results/run_meta.json`, `results/token_cost_output.txt` (stdout).

## Setup [source facts / settings]

| tokenizer | id | vocab | library | why it is in the comparison |
|---|---|---|---|---|
| gpt2 | tiktoken `gpt2` | 50,257 | tiktoken 0.14.0 | the v0 baseline; English-centric byte-level BPE |
| o200k | tiktoken `o200k_base` | 200,019 | tiktoken 0.14.0 | modern general-purpose multilingual LLM tokenizer; my **stand-in for the production tokenizer**, since FLM-4B's (128k vocab) is not available |
| sarvam1 | HF `sarvamai/sarvam-1` @ `e9607337286ddf496d4a2562b194e489dcf3feea` | 68,096 | tokenizers 0.23.3 | Indic-specialised LLM tokenizer (10 Indic langs + en). This is the "route Indic to an Indic model" option. **License: Sarvam AI Research License, non-commercial.** Fine for this measurement; production use needs permission. |

Text handling is identical for all three tokenizers:
- outer whitespace stripped, nothing else;
- no lowercasing, no NFC; NFC and collapsed internal whitespace are reported as sensitivities;
- **no special tokens**: tiktoken `encode_ordinary`; `tokenizers` `add_special_tokens=False`, so
  sarvam-1's BOS is excluded.

sarvam-1 ids from `tokenizers` match `transformers.AutoTokenizer` on all 8,096 sentences (0
mismatches, NOTEBOOK entry "A3 corrected analysis").

Denominators (all nonzero for every sentence; the script exits otherwise):

| denominator | computed as | what it holds constant |
|---|---|---|
| **parallel sentence** | line *i* of each FLORES file | **the meaning**: same source sentence, professionally translated |
| whitespace word | `len(text.split())` | an orthographic unit; words per meaning vary 14.8–25.3 across these languages |
| grapheme | `regex` `\X` (extended grapheme clusters, UAX #29 incl. GB9c) | a visual character; definition varies by script (conjunct handling) |
| UTF-8 byte | `len(text.encode())` | storage; 1 B per Latin char vs 3 B per Indic code point |

Aggregation is **pooled** (Σtokens / Σunits), because cost adds up tokens. Sentence-level
distributions are reported alongside so long lines can't dominate silently. The CI is a paired
bootstrap over sentence indices (1,000 reps, seed 0).

## Results [measured]

`premium` = Σ tokens(L) / Σ tokens(eng) over the same 1,012 sentences, same tokenizer.
`index@o200k` = tokens per sentence ÷ English tokens per sentence **on o200k**. It is a
normalised *token-count* index, so token counts can be compared across tokenizers on one scale. It
is **not** a cost: different models have different per-token costs. o200k is not our production
tokenizer either.

| tok | lang | tok/sent | tok/word | tok/grapheme | tok/byte | **premium** [95% CI] | sentence median (IQR) | premium, NFC input | index@o200k |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | eng | 26.7 | 1.23 | 0.205 | 0.205 | **1.00** | 1.00 | 1.00 | 1.01 |
| gpt2 | hin | 198.1 | 7.82 | 2.332 | 0.595 | **7.41** [7.32–7.50] | 7.40 (6.53–8.38) | 7.42 | 7.46 |
| gpt2 | kan | 363.0 | 22.82 | 4.066 | 0.979 | **13.59** [13.42–13.74] | 13.50 (11.84–15.50) | 13.58 | 13.67 |
| gpt2 | tam | 415.2 | 25.05 | 4.213 | 0.997 | **15.54** [15.34–15.71] | 15.60 (13.64–17.72) | 15.54 | 15.64 |
| gpt2 | tel | 346.6 | 20.71 | 4.581 | 0.992 | **12.97** [12.79–13.12] | 12.95 (11.37–15.00) | 12.97 | 13.05 |
| gpt2 | mal | 405.1 | 27.46 | 5.162 | 0.996 | **15.16** [14.97–15.34] | 15.14 (13.30–17.08) | 15.16 | 15.26 |
| gpt2 | ben | 255.4 | 13.25 | 3.164 | 0.750 | **9.56** [9.44–9.66] | 9.52 (8.40–10.78) | 9.61 | 9.62 |
| gpt2 | mar | 210.0 | 11.16 | 2.646 | 0.598 | **7.86** [7.76–7.94] | 7.83 (7.00–8.82) | 7.86 | 7.91 |
| o200k | eng | 26.6 | 1.23 | 0.204 | 0.203 | **1.00** | 1.00 | 1.00 | 1.00 |
| o200k | hin | 41.8 | 1.65 | 0.492 | 0.125 | **1.57** [1.56–1.59] | 1.57 (1.41–1.75) | 1.57 | 1.57 |
| o200k | kan | 52.3 | 3.29 | 0.585 | 0.141 | **1.97** [1.95–1.99] | 1.96 (1.74–2.19) | 1.97 | 1.97 |
| o200k | tam | 52.6 | 3.17 | 0.533 | 0.126 | **1.98** [1.96–2.00] | 1.96 (1.74–2.22) | 1.98 | 1.98 |
| o200k | tel | 51.2 | 3.06 | 0.677 | 0.147 | **1.93** [1.91–1.95] | 1.93 (1.70–2.18) | 1.93 | 1.93 |
| o200k | mal | 52.0 | 3.52 | 0.662 | 0.128 | **1.96** [1.94–1.98] | 1.93 (1.72–2.19) | 1.96 | 1.96 |
| o200k | ben | 45.1 | 2.34 | 0.558 | 0.132 | **1.70** [1.68–1.71] | 1.69 (1.53–1.89) | 1.71 | 1.70 |
| o200k | mar | 48.2 | 2.56 | 0.608 | 0.137 | **1.82** [1.80–1.83] | 1.80 (1.62–2.00) | 1.82 | 1.82 |
| sarvam1 | eng | 31.0 | 1.43 | 0.238 | 0.238 | **1.00** | 1.00 | 1.00 | 1.17 |
| sarvam1 | hin | 35.5 | 1.40 | 0.418 | 0.107 | **1.14** [1.13–1.15] | 1.14 (1.02–1.27) | 1.13 | 1.34 |
| sarvam1 | kan | 37.7 | 2.37 | 0.423 | 0.102 | **1.22** [1.20–1.23] | 1.20 (1.07–1.36) | 1.21 | 1.42 |
| sarvam1 | tam | 36.0 | 2.17 | 0.365 | 0.086 | **1.16** [1.15–1.17] | 1.14 (1.03–1.29) | 1.16 | 1.35 |
| sarvam1 | tel | 35.8 | 2.14 | 0.473 | 0.102 | **1.15** [1.14–1.17] | 1.13 (1.00–1.30) | 1.15 | 1.35 |
| sarvam1 | mal | 42.1 | 2.86 | 0.537 | 0.103 | **1.36** [1.34–1.37] | 1.34 (1.20–1.52) | 1.36 | 1.59 |
| sarvam1 | ben | 39.8 | 2.07 | 0.493 | 0.117 | **1.28** [1.26–1.30] | 1.26 (1.08–1.47) | **1.12** | 1.50 |
| sarvam1 | mar | 33.3 | 1.77 | 0.419 | 0.095 | **1.07** [1.06–1.08] | 1.06 (0.96–1.18) | 1.07 | 1.25 |

Collapsing internal whitespace changes no premium by more than 0.03 (column in `summary.csv`).

## What the numbers say [interpretation of measured results]

1. **The tokenizer, not the script, drives the cost.** For identical Hindi text the premium is 7.41×,
   1.57× and 1.14× across the three tokenizers. gpt2 spends about **1 token per UTF-8 byte** on the
   Dravidian scripts (tok/byte 0.98–1.00), which means it is effectively byte fallback with no
   vocabulary for these scripts.
2. **On a modern multilingual tokenizer (o200k) the same content costs 1.6–2.0× English.** Hindi
   is cheapest at 1.57; the four Dravidian languages cluster at 1.93–1.98. That is far below the
   "6×" in REPORT_v0.
3. **The Indic-specialised tokenizer saves 12–32% of Indic tokens against o200k:** hin 41.8 → 35.5,
   tam 52.6 → 36.0, mar 48.2 → 33.3. It costs **+17% on English** (26.6 → 31.0). Routing by
   language is therefore not free for English. For Bengali the saving depends on normalisation:
   1.28 raw vs 1.12 with NFC, because precomposed য় falls back to bytes (A2-N2).
4. **The other denominators give wrong answers to the cost question.**
   - **Per word** puts the Dravidian languages above Hindi by more than the cost difference
     (o200k: mal 3.52 vs hin 1.65 tok/word, a 2.1× gap; per sentence the gap is 52.0 vs 41.8, 1.24×).
   - **Per byte inverts the sign.** Hindi on o200k is 0.125 tok/byte vs English 0.203, which reads
     as "38% cheaper", while the same sentences cost 1.57× more.
   - **Per grapheme** overstates the gap (hin/eng 2.41× vs 1.57×).
5. **Per-request variation is wide.** Sentence-level IQRs are about ±10–15% around the median, e.g.
   o200k Hindi 1.41–1.75. Budget with the pooled number; set alerting thresholds with the spread.

## The single number for routing and cost

**The same-content token premium: tokens per parallel sentence for language L ÷ the same for
English, both on the production tokenizer** (`premium_vs_base`). It must be measured on the
production tokenizer, which was not available here, so the o200k numbers are an interim proxy.
The secondary column `index_vs_ref` uses the same denominator but compares *token counts* across
tokenizers. Turning it into a routing *cost* comparison needs each model's per-token cost
(point 3 below).

**Why this denominator.** A routing or cost decision compares *the same request* served in
different ways: in Hindi vs English, or on tokenizer A vs B. The denominator must therefore hold
**the request's content** constant. A professionally translated parallel sentence is the closest
measurable proxy for a fixed unit of meaning.

Words, graphemes and bytes are *language-internal* units:
- their count per unit of meaning varies by language: 14.8–25.3 words, 75.7–130.4 graphemes and
  130–417 bytes per sentence here;
- dividing by them mixes tokenizer efficiency with morphology and orthography;
- they can rank languages differently (per word) or flip the sign (per byte).

On a fixed model, serving cost scales with token counts: prefill FLOPs, KV memory per token (B1:
114,688 B/token on FLM-4B) and decode steps. So for one model, the same-content premium is the cost
multiplier for a language's requests. Across different models it is only the token half of the
cost; the other half is each model's per-token cost.

**What it can predict.** The token multiplier for the same content, and therefore relative prefill
compute, KV-cache footprint (how many concurrent sequences fit), context-window headroom, and for
equal-meaning outputs, the number of decode steps.

**What it cannot predict:**
1. **Real traffic content.** Chat is short, code-mixed and sometimes romanised; FLORES is formal
   translated prose (CORPUS.md limits).
2. **Response length.** Models may answer at different lengths per language, and decode tokens are
   the expensive ones.
3. **Price per token on different models.** A 2B Indic model's token is cheaper than a 4.2B
   model's token; the full routing cost is tokens × per-token cost per model.
4. **Quality.** A cheaper route is useless if answers degrade.
5. **The production tokenizer itself.** o200k is a proxy. Rerun with one flag once FLM-4B's
   tokenizer is available:
   `token_cost.py --tokenizers hf:<flm-repo>,sarvam1 --ref-tokenizer hf:<flm-repo>`
