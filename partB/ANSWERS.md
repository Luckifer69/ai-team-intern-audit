# Part B — Capacity reconciliation

Inputs: `starter_kit/bench/model_spec.md`, `starter_kit/bench/bench_log.csv` (unchanged; sha256 in
`STARTER_KIT_SHA256.txt`). Every number below is printed by `python3 -I partB/capacity.py`
(stdlib only; full output in `capacity_output.txt`, machine-readable in `results.json`) and was
re-derived independently with `awk` (NOTEBOOK entry "Part B script + independent check"). "CSV line N" = physical line in the file,
header = line 1. No GPU was available, so nothing here was re-benchmarked.

Evidence labels: **[S]** source fact (spec/CSV) · **[C]** calculation from [S] · **[A]** assumption ·
**[H]** hypothesis · **[F]** forecast (not measured).

---

## B1 — KV-cache bytes per token and concurrent 4096-token sequences

**(a) Bytes per token, exactly [C]**

    KV bytes/token = 2 (K and V) × layers × kv_heads × head_dim × bytes/element
                   = 2 × 28 × 8 × 128 × 2 B (fp16)
                   = 114,688 B  = 112 KiB per token

GQA matters: KV uses the 8 KV heads, not the 24 query heads (24×128 = 3072 = d_model, consistent [S]).

**(b) Max concurrent 4096-token sequences [C, with A1–A3]**

    usable GPU memory  = 24 GB × 0.92            = 22.08 GB
    weights            = 4.2e9 params × 2 B      =  8.40 GB
    non-KV overhead    = 1.6 GB (given)
    KV pool            = 22.08 − 8.40 − 1.6      = 12.08 GB  →  12.08e9 / 114,688 = 105,329 tokens
    one 4096-token seq = 4096 × 114,688 B        = 469,762,048 B (448 MiB, 0.470 GB)
    concurrent seqs    = 12.08e9 / 469,762,048   = 25.7  →  **25 full-length sequences**

- [A1] "GB" in the spec is decimal (1e9 B). Reading 24 GB and 1.6 GB as GiB instead gives 118,497
  tokens → 28 sequences. The log favours the decimal reading (next point).
- [A2] Weight memory = parameter count × 2 B (fp16); no quantisation, no tied-embedding savings.
- [A3] KV is allocated in fixed blocks (vLLM-style). 4096 divides evenly into 16-token blocks, so no
  rounding loss.

**Check against the log [S → C].** Implied capacity = `batch_size × (prompt_len + gen_len) / kv_cache_util`:

| CSV line | batch | prompt | peak tokens | kv_cache_util | implied capacity | preempted_seqs | batch − 25 |
|---|---|---|---|---|---|---|---|
| 10 | 8 | 3584 | 32,768 | 0.31 | 105,703 | 0 | — |
| 11 | 16 | 3584 | 65,536 | 0.62 | 105,703 | 0 | — |
| 12 | 24 | 3584 | 98,304 | 0.93 | 105,703 | 0 | — |
| 13 | 32 | 3584 | 131,072 | 0.97 | (saturated) | **7** | **7** |
| 14 | 48 | 3584 | 196,608 | 0.97 | (saturated) | **23** | **23** |

- The rows with the most precise utilisation imply about 105.7k tokens of KV capacity. My spec-only
  estimate is 105,329, a 0.4% gap, which is within the 2-decimal rounding of `kv_cache_util`.
- Low-utilisation rows (0.01–0.03) give 77k–154k only because of that rounding.
- Batch 24 (98,304 tokens) fits with no preemption.
- Batches 32 and 48 need 131k and 197k tokens. Utilisation pins at 0.97, and `preempted_seqs` equals
  exactly batch − 25 in both rows.

This is strong consistency with the 25-sequence prediction. It is not proof: the CSV reports peak
utilisation and a preemption count, not the scheduler's state over time.

## B2 — The long-context anomaly

**What [S → C].** Rows with `prompt_len=3584` (CSV lines 9–14):

| line | batch | wall_clock_s | reported_tok_s | naive linear from b=24 | generated tok/s | itl_ms_p50 | preempted_seqs | kv_cache_util |
|---|---|---|---|---|---|---|---|---|
| 12 | 24 | 61.16 | 1607.4 | 1607.4 | 200.9 | 96.07 | 0 | 0.93 |
| 13 | 32 | 94.71 | 1384.0 | 2143.2 | 173.0 | 101.79 | 7 | 0.97 |
| 14 | 48 | 151.41 | 1298.5 | 3214.8 | 162.3 | 100.00 | 23 | 0.97 |

Past batch 24, throughput does not just saturate; it **falls**.

- Compared with batch 24, `reported_tok_s` and generated tok/s both drop 13.9% at batch 32 and 19.2% at batch 48.
- Against the linear extrapolation the report used, batch 32 is 35% short and batch 48 is 60% short.
- `wall_clock_s` grows 1.55× for 1.33× the work (b=32) and 2.48× for 2.00× the work (b=48).
- The short-prompt sweep keeps scaling to batch 64 (lines 2–8: 0 preemptions, utilisation ≤ 0.47).

**Mechanism [H, supported by three observations].** The KV pool holds only about 25 sequences at
4096 tokens (B1). Beyond that, the scheduler preempts sequences as their KV grows during decode. A
preempted sequence has its KV dropped (recompute mode, vLLM's default) or swapped out, and its
3,584-token prefill is redone later. The extra requests effectively run in a second wave, plus
recompute overhead.

1. `preempted_seqs` = batch − 25 exactly (7 and 23), and `kv_cache_util` is pinned at 0.97.
2. ITL plateaus at about 100 ms. Decode is memory-bandwidth-bound. Each step reads the weights plus
   the KV of every running sequence.
   - Calibration on line 12: 8.40 GB + 24 × 3840 × 114,688 B = 18.97 GB per step. That is 63.2 ms at
     300 GB/s; the observed 96.07 ms means about 66% effective bandwidth.
   - At the same efficiency the model predicts ITL of 98.3 ms if 25 sequences run, 113.9 ms if 32
     run, and 149.6 ms if 48 run.
   - Observed: 101.79 and 100.00 ms. That is consistent with roughly 25 running, not 32 or 48 [C].
3. The wall clock is consistent with waves. For 32 requests, b=24 (61.16 s) + b=8 (36.30 s) ≈ 97.5 s
   vs 94.71 s observed. For 48 requests, two b=24 waves ≈ 122.3 s vs 151.41 s observed. The extra
   ~29 s is attributed to recompute and preemption churn [H].

**Proposed change.** Cap admission at the KV fit for the 4k-context pool:
`max_num_seqs=24` (vLLM), and scale with replicas rather than batch.

Forecast, using only measured rows as building blocks [F]:
- **Sustained load** (≥24 requests in flight): about 200.9 generated tok/s per L4, 0.39 req/s, and 0
  preemptions (the measured batch-24 row). That is +16% over batch 32 and +24% over batch 48.
- **Closed burst of 48:** about 122 s instead of 151 s (+24% goodput).
- **Closed burst of 32:** about 97.5 s instead of 94.7 s, roughly −3%. So the cap does **not** help a
  mild overload; it pays off as overload grows.
- Queued requests wait instead of thrashing, so TTFT for the second wave rises by about one wave
  (~61 s).

Assumptions: arrivals are like the test (3,584 in / 512 out, no early stop); a second wave costs what
the measured smaller-batch rows cost.

An alternative not chosen as "the" change: `kv_cache_dtype=fp8` halves KV bytes per token, giving
about 210k tokens and 51 sequences of 4096 [C]. That should remove preemption at b=32 and b=48, but its
throughput and quality effects cannot be anchored to any row in this log.

## B3 — The misread column and honest goodput

**The misreading [S → C].** `reported_tok_s` counts prompt **and** generated tokens:
`reported_tok_s = (prompt_len + gen_len) × num_requests / wall_clock_s` holds on **all 13 rows**
(max deviation 0.022%). On long-prompt rows, 3584 / 4096 = 87.5% of the counted tokens are prompt
tokens. That is prefill work the user does not wait to read, so a bigger number here mostly means
"more prompt was processed". Generated-token goodput is the rate at which output tokens are produced.

**Goodput, batch 24, prompt_len 3584 (CSV line 12), two independent ways [C]:**

1. From the wall clock: `num_requests × gen_len / wall_clock_s = 24 × 512 / 61.16 s` = **200.92 tok/s**
2. From the counter: `reported_tok_s × gen_len / (prompt_len + gen_len) = 1607.4 × 512/4096 = 1607.4 × 0.125`
   = **200.93 tok/s**

The two agree to 0.005%, which is CSV rounding. Way 1 uses `wall_clock_s` and never touches the
counter. Way 2 uses only the counter and the token composition.

Sanity bound: steady-state decode rate = batch / ITL = 24 / 0.09607 s = 249.8 tok/s. Goodput is 80%
of that; prefill and ramp-down take the remaining wall time.

I deliberately do **not** use `e2e_ms_p95`. In 12 of 13 rows it exceeds `wall_clock_s` (ratio
1.03–1.18), which is impossible if both timers share a start point. Likewise `ttft_ms_p50` ≈ 0.5 s at
batch 24 implies more prefill compute than the stated 121 TFLOPS peak allows (NOTEBOOK entry "First look at bench_log.csv semantics"). Both are
flagged as data-quality questions and are not used.

**Both conclusions in REPORT_v0 §2 fail [C]:**
- **"Longer prompts give better throughput."** At batch 16, long prompts deliver 163.9 generated
  tok/s vs 294.5 for short prompts (−44%). The counter rose only because prompt tokens are counted.
- **"Batch 48 should reach ~3200 tok/s."** That is 1607.4 × 48/24 = 3214.8. But the log already
  contains batch 48 (line 14): 1298.5 counter tok/s and 162.3 generated tok/s. Linear scaling also
  ignores the KV ceiling at ~25 sequences.

**What the report should have said.** For 3.5k-in / 512-out traffic, one L4 sustains about **200
generated tok/s (≈0.39 req/s)** at ≤24 concurrent requests. Going past about 25 concurrent requests
causes KV preemption and *lowers* goodput (173 at batch 32, 162 at batch 48). Plan capacity by adding
replicas at batch ≈24. Report generated-token goodput separately from prompt-token throughput, and
don't extrapolate linearly past the KV limit.

## B4 — Confirmatory counter

I would pull one metric: the scheduler's running-batch gauge, **`vllm:num_requests_running`**. I'm
assuming a vLLM stack because the spec's parameter names match it [A]. I'd sample it at ≥1 Hz while
re-running the batch-32 and batch-48 long-prompt loads. If the KV-capacity mechanism is right [F], it
peaks at no more than about 29 while sequences are near prompt length (105,329 / 3,584 = 29.4). It then
settles at about **25** during decode (105,329 / 4,096 = 25.7), and never reaches 32 or 48. That
discriminates against the competing explanation, where bandwidth or compute saturates with *all*
sequences resident: that would predict the gauge reads 32 / 48 (and ITL near 114 / 150 ms). A reading of
32 or 48 would therefore refute my B2 mechanism. (`preempted_seqs` is already in the CSV, so
pulling preemptions again adds nothing.)
