## Memo: tokenizer cost audit (corrects REPORT_v0 §1)

**Bottom line.** Treat REPORT_v0's "6× for Hindi" as **unverified**: it is a per-word ratio on
GPT-2, not a measured per-request cost (`AUDIT.md`). Use the o200k figures below only as an interim
proxy until our production tokenizer is measured.

**Corrected headline [measured].** Tokens for the *same* 1,012 FLORES-200 sentences, as a multiple
of English tokens (95% CIs ±~1%):

| tokenizer (× English tokens) | hin | kan | tam | tel | mal | ben | mar |
|---|---|---|---|---|---|---|---|
| GPT-2 (REPORT_v0's) | 7.4 | 13.6 | 15.5 | 13.0 | 15.2 | 9.6 | 7.9 |
| o200k_base (proxy for ours) | **1.6** | **2.0** | **2.0** | **1.9** | **2.0** | **1.7** | **1.8** |
| sarvam-1 (Indic) ÷ o200k English* | 1.3 | 1.4 | 1.4 | 1.4 | 1.6 | 1.5 | 1.3 |

\*A token-count index, not a cost: models differ in per-token cost.
- **The tokenizer drives token count, not the script.** The same Hindi text takes 7.4×, 1.6× or
  1.1× English's tokens depending on the tokenizer.
- **The metric was wrong.** Per-word and per-character ratios divide by units that carry different
  amounts of meaning in different languages. From the same token counts, per word makes Hindi on
  sarvam-1 look *cheaper* than English (0.98×); the same sentences actually take 1.14×. The right scalar is **tokens per
  parallel sentence relative to English**: it holds the request's meaning constant.
- `fertility.py` details shift results by ≤3.5%: a double-space bug, and lowercasing (intentional, but
  production text keeps its case).

**Routing recommendation.**
1. **Interim planning figure only:** ~1.6× (Hindi) to ~2.0× (Dravidian) English tokens on a modern
   tokenizer [assumption: ours behaves like o200k]. Measure ours before any budget decision (Monitor).
2. **An Indic-specialised route saves 12–32% of Indic tokens; it is not a 6× fix.** It *adds* 17% on
   English, so if adopted, route per detected language. Adopt only if tokens × per-token cost beats
   the main model, Indic quality is at parity on a native eval, and licensing is cleared
   (sarvam-1's license is non-commercial).
3. NFC-normalise input before an Indic tokenizer (raw Bengali: +14.6% on sarvam-1).

**Biggest caveat.** FLORES is formal, translated prose and covers only prompts. Real chat is short,
code-mixed and often romanised, and reply lengths may differ by language; the CIs cover FLORES only.

**Monitor (proposed; needs the deployed tokenizer): same-content token premium**
P_L = Σᵢ T(L,i) ÷ Σᵢ T(eng,i). T = deployed-tokenizer tokens (text as written plus the serving
stack's normalisation, if any; no special tokens). i runs over the fixed 1,012 aligned FLORES-200
devtest sentences (eng + 7 Indic; sha256 in `manifest.json`). Run `token_cost.py --tokenizers
hf:<deployed>` and read `premium_vs_base`. Baseline = its first run (o200k's 1.57–1.98 are
placeholders). Recompute after any tokenizer, model or serving-stack change. Inputs are fixed, so
any move is a tokenizer/preprocessing change, not traffic, and triggers a cost re-plan.
