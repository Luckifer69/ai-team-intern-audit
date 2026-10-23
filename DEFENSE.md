# Live-defense sheet

This is reasoning to understand, not a script. Every number here is in a saved output file;
re-derive the starred ones by hand.

## 1. Equations to re-derive

**B1 — KV bytes per token\*.** `2 (K,V) × layers × kv_heads × head_dim × bytes = 2 × 28 × 8 × 128 × 2 = 114,688 B` (112 KiB).
Use the 8 KV heads, not the 24 query heads; GQA is the reason. With full multi-head attention (24 KV
heads) it would be 344,064 B/token and only 8 sequences would fit.

**B1 — capacity\*.**
- Usable memory: 24 × 0.92 = 22.08 GB.
- KV pool: 22.08 − 8.4 (weights, 4.2e9 × 2 B) − 1.6 = **12.08 GB**.
- 12.08e9 / 114,688 = **105,329 tokens**; / 4096 = 25.7 → **25 sequences**.
- Log check: rows at batch 8/16/24 imply 105.7k tokens (`batch × 4096 / kv_cache_util`), and
  `preempted_seqs` = batch − 25 (7 and 23).

**B3 — goodput, two ways\*** (CSV line 12):
1. `24 × 512 / 61.16 = 200.92 tok/s`
2. `1607.4 × 512/4096 = 200.93 tok/s`

The `reported_tok_s` formula, `(prompt_len + gen_len) × num_requests / wall_clock_s`, holds on all
13 rows within 0.022%.

**B2 — decode memory model.**
- Bytes per decode step = weights + running_seqs × context × 114,688.
- Calibrated at b=24: 18.97 GB/step, i.e. 63.2 ms at 300 GB/s, vs 96.07 ms observed = 66% efficiency.
- Predicted ITL: 98.3 ms with 25 sequences running, 113.9 ms with 32, 149.6 ms with 48.
- Observed ITL is ~100 ms, so ~25 sequences are actually running.

**A3 — the scalar.** `premium(L) = Σ_i tokens_L[i] / Σ_i tokens_eng[i]` over the same parallel
sentences. `index_vs_ref` divides by English tokens on a fixed reference tokenizer, so values are
comparable across tokenizers. It is a token-count index; it becomes a cost only with each model's
price per token. The routing scalar is the premium on the production tokenizer.

**A2 E5 — why it isolates the denominator\*.** Same token counts T on both sides:
`per-word ratio ÷ premium = [(T_L/W_L)/(T_eng/W_eng)] ÷ (T_L/T_eng) = W_eng/W_L`. The tokenizer cancels.
Use word **totals** (E1 output): Hindi 21,901 / 25,643 = 0.854 → −14.6% for every tokenizer; Malayalam
21,901 / 14,930 = 1.467 → +46.7%. (Rounded words/sentence, 21.6/14.8, gives 45.9%, so quote totals.)
E5 part 2 (v0 headline 6.11 → 7.41) changes four settings at once and is labelled *combined*.

**B4 — the one metric.** `vllm:num_requests_running` during the batch-32/48 rows. Expect ≈25 in decode
(≤29 while sequences are near prompt length: 105,329 / 3,584). If it shows 32/48, the KV-capacity explanation is wrong.

## 2. What the benchmark columns mean

- **`reported_tok_s`** counts prompt **and** generated tokens per wall-clock second. In long rows,
  87.5% of it is prompt, so it rises with prompt length even while useful output falls.
- **Goodput** = generated tokens per second, `num_requests × gen_len / wall_clock_s`.
- **`kv_cache_util`** is *peak* block usage at 2 decimals; that rounding is why the low-utilisation
  rows imply noisy capacities.
- **`itl_ms_p50`** is per-request decode step time; aggregate decode rate = batch / ITL.
- **Not trusted:** `e2e_ms_p95` exceeds `wall_clock_s` in 12/13 rows, and TTFT ≈ 0.5 s at b=24 needs
  more than peak FLOPs. I say so and don't build on either.

## 3. Tokenizer methodology in one breath

The question is: what does the *same request* cost in another language, or on another tokenizer?
The denominator therefore has to hold the **content** fixed, and a professionally translated
parallel sentence does that.

- Words vary 14.8–25.3 per sentence across languages: agglutinative Dravidian vs analytic Hindi.
- Bytes vary 130–417 per sentence: UTF-8 uses 3 B per Indic code point.
- Graphemes depend on script-specific conjunct rules.

Measured consequences: per word reorders languages and flips sarvam-1 Hindi (0.98 vs 1.14, same token counts); per
byte flips the sign (Hindi "38% cheaper" on o200k vs actually 1.57×). Settings: no lowercasing,
no special tokens, strip only, pooled sums; NFC and whitespace collapse are measured as
sensitivities.

## 4. Part C assumptions to own

- We control a ~4B open model.
- The reviewer manages ~40 pairwise judgments/h (the biggest lever: at 20/h the budget halves to 600).
- Real prompts exist; prefix caching is on; A100 at ~312 TFLOPS × 35% (training only).
- **S1:** serving runs on Part B's L4. Part C names no serving GPU, so every latency/KV number for
  (b) and (c) is conditional on that.
- The paid reviewer reads only hi + kn: 1,200 judgments. Prompt path 580; with the Day-8 switch to
  SFT, 880. A 6-language SFT would need 900 QA judgments, 600 of them in languages the reviewer can't
  read, so the SFT fallback is hi/kn only.
- Day 1 = 2026-10-22 because that is the date the memo was written; no project start date was given.

The decision follows from the binding constraint, which is 30 reviewer-hours, not GPU time.

## 5. Rerun and modify live

```bash
./run_all.sh                                                    # everything, ~90 s cold
.venv/bin/python -I partA/audit/audit_fertility.py --only E1 E7  # one experiment
.venv/bin/python -I partA/scripts/token_cost.py --corpus eng=a.txt --corpus hin=b.txt --tokenizers gpt2,o200k --out-dir /tmp/x
.venv/bin/python -I partA/scripts/token_cost.py --tokenizers o200k,hf:xlm-roberta-base --out-dir /tmp/x   # any HF tokenizer.json
python3 -I partB/capacity.py --goodput-batch 16 --goodput-prompt 512 --out /tmp/x.json    # another row
python3 -I partB/capacity.py --gpu-mem-util 0.95 --kv-bytes 1 --seq-len 2048 --out /tmp/x.json   # what-ifs
python3 -I partB/capacity.py --csv new_log.csv --out /tmp/x.json   # same columns; missing REPORT rows are skipped
```

Without `--out-dir` / `--out`, token_cost.py and capacity.py overwrite the shipped results
(`partA/results/`, `partB/results.json`); `./run_all.sh` restores them. The audit's `--only` never writes.

Exporting `HF_HOME=$PWD/.hf_cache TIKTOKEN_CACHE_DIR=$PWD/.tiktoken_cache` keeps downloads local.
Where to change things:
- **New tokenizer:** add it to `TOKENIZERS` in `partA/scripts/common.py`.
- **New denominator:** add it to `units()` in the same file.
- **New audit switch:** add a parameter to `analyze_variant()` plus an `E*` function.

## 6. Likely questions and answer outlines

1. **"Is `split(" ")` really a bug?"** An empty string is not a word. Minimal case: `"books  in"` → 8
   vs 7 words. Effect is ≤2.2% (Kannada has 213 double-space lines) and changes no ranking. Real
   but minor.
2. **"Why isn't `random.seed` a bug?"** Nothing samples. Deleting both lines gives byte-identical
   output (E6). It's dead code.
3. **"Why not per-character?"** The same tokens give 2.65×, 7.00× or 10.86× depending on what
   "character" means (E4). A metric that changes with a definition carries no cost information.
4. **"Your premium uses FLORES; real traffic differs."** Agreed. The CIs only cover sampling within
   FLORES. The memo's monitor deliberately uses *fixed* FLORES inputs on the deployed tokenizer, so
   it isolates tokenizer/preprocessing changes from traffic mix. It does not measure the chat-domain
   premium. That needs an aligned chat-style set, which this submission doesn't have (caveat in memo).
5. **"Why o200k as the production proxy?"** The real 128k tokenizer isn't available, and o200k is a
   modern multilingual BPE of similar scale. It's a one-flag rerun when ours is available; I don't
   claim it equals ours.
6. **"Why is NFC not 'harmless'?"** It decomposes precomposed nukta letters. On sarvam-1 that moves
   Bengali by 14.6% because the precomposed form falls back to bytes. It is a correct normalisation,
   but it must match production preprocessing.
7. **"Two independent goodput derivations — they share the CSV?"** Way 1 uses only `wall_clock_s`;
   way 2 uses only `reported_tok_s`. They agree to 0.005%, which also confirms the counter's
   definition. A third check, the decode bound batch/ITL = 249.8, sits above both, as it must.
8. **"Why does throughput *drop* past batch 24?"** KV holds ~25 sequences of 4096 tokens. Beyond that
   the scheduler preempts (7 and 23 sequences) and recomputes or swaps. Evidence: preempted =
   batch − 25, utilisation pinned at 0.97, and ITL flat at ~100 ms, matching 25 running.
9. **"Does capping max_num_seqs help?"** Under sustained load: ~200.9 vs 162–173 gen tok/s. For a
   48-request burst, +24%. For a 32-request burst, −2.8%. This is a forecast assembled from measured
   rows, not a measurement.
10. **"Why prompt-only for Part C if SFT has the higher ceiling?"** The GPU needs ~3 h; the binding
    constraint is 30 reviewer-hours (1,200 judgments, hi/kn only). SFT on synthetic text in the 4
    languages nobody can review carries unmeasurable risk. Prompting can be measured this week; the kill date switches to SFT while the
    GPU is still there.
11. **"Your notebook timestamps?"** The AI typed invented times into headings 2–15. Caught at 21:36
    (Entry 16). After the external review the headings became numbered entries: a clock time appears only
    where it was read from `date`, and Entries 2–15 say "time not recorded". The invented times are listed
    openly in Entry 20 (AI_USAGE #1). Content and order are real.
12. **"Is E5 isolated?"** The first version wasn't (review caught it; AI_USAGE #10). Now part 1 uses
    identical token counts, so the distortion is W_eng/W_L and the same for all three tokenizers.

## 7. Counterfactuals

- **gpu_memory_utilization 0.95:** pool 12.80 GB → 27 sequences (0.90 → 24). Preemption then starts
  just above batch 27.
- **Prompt 2048 + gen 512 (2560 tokens):** 41 sequences fit, so batch 32 would not preempt.
- **FP8 KV cache:** 57,344 B/token → 51 sequences. Batches 32 and 48 should not preempt; the
  throughput gain is unknown without a run.
- **max_model_len 8192:** 12 full sequences.
- **Corpus becomes chat-like:** hypothesis only. 3 close short sample pairs (E10) gave o200k 1.39 vs
  FLORES 1.57, so the premium could be lower; romanised Hindi could push it higher. Measure, don't
  guess.
- **Production tokenizer behaves like GPT-2:** Indic premium becomes 7–16×, and an Indic route would
  save 80%+ of tokens. Routing then becomes the big lever.
- **Reviewer also reads Tamil:** Tamil moves into the validated launch set; the decision logic is
  unchanged.
- **Day-8 win rate 65%:** between kill (60%) and success (70%), so the prompt path continues. Spend
  week-2 reviewer hours on the full 100-prompt eval; launch requires ≥70%.

## 8. Risks to understand personally

1. **Whether the samples are parallel (finding D) depends on the Hindi glosses.** Read them
   yourself before the session.
2. **The production tokenizer is not measured.** Every "1.6–2.0×" is an o200k proxy.
3. **Part B rests on decimal GB and a vLLM-like scheduler.** The GiB reading gives 28 sequences, but
   the log matches 25.
4. **The Part C reviewer rate (40/h) is assumed.** The whole Part C plan scales with it.
5. **The AI wrote this.** Know E0's purpose (harness fidelity) and be able to rewrite any one
   experiment from memory.
