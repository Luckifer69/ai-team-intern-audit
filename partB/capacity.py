#!/usr/bin/env python3
"""
capacity.py -- Part B arithmetic, derived only from bench/model_spec.md values and bench_log.csv.

Standard library only. Every number in partB/ANSWERS.md is printed by this script.

Usage:
    python -I partB/capacity.py                          # B1-B3 for the default rows
    python -I partB/capacity.py --goodput-batch 16 --goodput-prompt 3584
    python -I partB/capacity.py --seq-len 2048 --gpu-mem-util 0.90 --kv-bytes 1   # what-ifs
    python -I partB/capacity.py --csv path/to/other_log.csv
"""

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTER = HERE.parents[1] / "starter_kit"

# ---- bench/model_spec.md (transcribed; see ANSWERS.md for the table) ----
SPEC = dict(params=4.2e9, layers=28, d_model=3072, q_heads=24, kv_heads=8, head_dim=128,
            weight_bytes=2, kv_bytes=2,                       # fp16 weights, fp16 KV
            gpu_mem_gb=24, gpu_mem_util=0.92, overhead_gb=1.6, max_model_len=4096,
            mem_bw_gbs=300, peak_tflops=121)


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):        # line 1 is the header
            r = {k: float(v) for k, v in r.items()}
            for k in ("batch_size", "prompt_len", "gen_len", "num_requests", "preempted_seqs"):
                r[k] = int(r[k])
            r["csv_line"] = i
            rows.append(r)
    return rows


def find(rows, batch, prompt, required=True):
    hit = [r for r in rows if r["batch_size"] == batch and r["prompt_len"] == prompt]
    if len(hit) != 1:
        if not required:
            return None
        raise SystemExit(f"expected exactly one row with batch_size={batch}, prompt_len={prompt}; got {len(hit)}")
    return hit[0]


def gen_goodput(r):
    return r["num_requests"] * r["gen_len"] / r["wall_clock_s"]


def b1(s, rows, seq_len):
    print("== B1  KV-cache bytes per token and concurrent sequences")
    assert s["q_heads"] * s["head_dim"] == s["d_model"], "spec inconsistent: q_heads*head_dim != d_model"
    kv_tok = 2 * s["layers"] * s["kv_heads"] * s["head_dim"] * s["kv_bytes"]
    print(f"  bytes/token = 2 (K,V) x {s['layers']} layers x {s['kv_heads']} kv_heads x {s['head_dim']} head_dim"
          f" x {s['kv_bytes']} B = {kv_tok:,} B = {kv_tok / 1024:g} KiB")
    weights = s["params"] * s["weight_bytes"]
    usable = s["gpu_mem_gb"] * 1e9 * s["gpu_mem_util"]
    kv_pool = usable - weights - s["overhead_gb"] * 1e9
    tokens = kv_pool / kv_tok
    per_seq = kv_tok * seq_len
    print(f"  usable = {s['gpu_mem_gb']} GB x {s['gpu_mem_util']} = {usable / 1e9:.2f} GB;"
          f" weights = {s['params'] / 1e9:g}e9 x {s['weight_bytes']} B = {weights / 1e9:.2f} GB;"
          f" overhead = {s['overhead_gb']} GB")
    print(f"  KV pool = {usable / 1e9:.2f} - {weights / 1e9:.2f} - {s['overhead_gb']} = {kv_pool / 1e9:.2f} GB"
          f" -> {tokens:,.0f} tokens")
    print(f"  one {seq_len}-token sequence = {per_seq:,} B = {per_seq / 2**20:g} MiB = {per_seq / 1e9:.4f} GB"
          f" -> {kv_pool / per_seq:.2f} -> max {int(kv_pool // per_seq)} concurrent sequences")
    # Unit-convention sensitivity: read "24 GB" and "1.6 GB" as GiB instead.
    usable_gib = s["gpu_mem_gb"] * 2**30 * s["gpu_mem_util"]
    pool_gib = usable_gib - weights - s["overhead_gb"] * 2**30
    print(f"  sensitivity (24 GiB / 1.6 GiB reading): KV pool {pool_gib / 2**30:.2f} GiB -> {pool_gib / kv_tok:,.0f} tokens"
          f" -> {int(pool_gib // per_seq)} sequences")

    print("  cross-check vs log: implied KV capacity = batch x (prompt+gen) / kv_cache_util (peak; util has 2 decimals)")
    print(f"    {'line':>4} {'batch':>5} {'prompt':>6} {'peak tok':>9} {'kv_util':>7} {'implied cap':>11}"
          f" {'pred util':>9} {'preempted':>9} {'batch-maxseq':>12}")
    max_seq = int(kv_pool // per_seq)
    check = []
    for r in rows:
        need = r["batch_size"] * (r["prompt_len"] + r["gen_len"])
        implied = need / r["kv_cache_util"] if r["kv_cache_util"] else float("nan")
        pred = min(need / tokens, 1.0)
        over = max(r["batch_size"] - max_seq, 0) if r["prompt_len"] + r["gen_len"] == seq_len else None
        check.append({"csv_line": r["csv_line"], "batch_size": r["batch_size"], "prompt_len": r["prompt_len"],
                      "peak_tokens": need, "kv_cache_util": r["kv_cache_util"], "implied_capacity": implied,
                      "predicted_util": pred, "preempted_seqs": r["preempted_seqs"], "predicted_overflow": over})
        print(f"    {r['csv_line']:>4} {r['batch_size']:>5} {r['prompt_len']:>6} {need:>9,} {r['kv_cache_util']:>7.2f}"
              f" {implied:>11,.0f} {pred:>9.2f} {r['preempted_seqs']:>9} {'' if over is None else over:>12}")
    return {"kv_bytes_per_token": kv_tok, "weights_bytes": weights, "usable_bytes": usable, "kv_pool_bytes": kv_pool,
            "kv_pool_tokens": tokens, "bytes_per_seq": per_seq, "max_concurrent_seqs": max_seq,
            "gib_reading_seqs": int(pool_gib // per_seq), "rows": check}


def b2(s, rows, prompt, kv):
    print(f"\n== B2  prompt_len={prompt} sweep vs naive scaling")
    sweep = sorted((r for r in rows if r["prompt_len"] == prompt), key=lambda r: r["batch_size"])
    if not sweep:
        raise SystemExit(f"no rows with prompt_len={prompt}; pass --anomaly-prompt with a prompt length in the CSV")
    peak = max(sweep, key=lambda r: r["reported_tok_s"])
    print(f"  {'line':>4} {'batch':>5} {'wall_s':>7} {'reported':>9} {'naive(lin)':>10} {'vs naive':>8}"
          f" {'gen tok/s':>9} {'ttft_p50':>8} {'itl_p50':>7} {'preempt':>7} {'kv_util':>7}")
    out = []
    for r in sweep:
        naive = peak["reported_tok_s"] * r["batch_size"] / peak["batch_size"]
        g = gen_goodput(r)
        out.append({"csv_line": r["csv_line"], "batch_size": r["batch_size"], "wall_clock_s": r["wall_clock_s"],
                    "reported_tok_s": r["reported_tok_s"], "naive_linear_from_peak": naive,
                    "gen_tok_s": g, "itl_ms_p50": r["itl_ms_p50"], "preempted_seqs": r["preempted_seqs"],
                    "kv_cache_util": r["kv_cache_util"]})
        print(f"  {r['csv_line']:>4} {r['batch_size']:>5} {r['wall_clock_s']:>7.2f} {r['reported_tok_s']:>9.1f}"
              f" {naive:>10.1f} {100 * (r['reported_tok_s'] / naive - 1):>7.1f}% {g:>9.1f} {r['ttft_ms_p50']:>8.1f}"
              f" {r['itl_ms_p50']:>7.2f} {r['preempted_seqs']:>7} {r['kv_cache_util']:>7.2f}")
    after = [r for r in sweep if r["batch_size"] > peak["batch_size"]]
    for r in after:
        print(f"  batch {r['batch_size']} vs {peak['batch_size']}: reported {100 * (r['reported_tok_s'] / peak['reported_tok_s'] - 1):+.1f}%,"
              f" generated {100 * (gen_goodput(r) / gen_goodput(peak) - 1):+.1f}%,"
              f" wall {r['wall_clock_s'] / peak['wall_clock_s']:.2f}x for {r['batch_size'] / peak['batch_size']:.2f}x the work")

    # Decode is memory-bound: bytes read per decode step = weights + KV of all running sequences.
    ctx = prompt + sweep[0]["gen_len"] / 2                      # mean context length during decode
    print(f"  decode-step memory model (mean context {ctx:.0f} tokens, peak BW {s['mem_bw_gbs']} GB/s):")
    eff = None
    for n_run in (peak["batch_size"], kv["max_concurrent_seqs"], 32, 48):
        byts = kv["weights_bytes"] + n_run * ctx * kv["kv_bytes_per_token"]
        t = byts / (s["mem_bw_gbs"] * 1e9) * 1e3
        if eff is None:
            eff = t / peak["itl_ms_p50"]
            print(f"    calibrate on batch {n_run}: {byts / 1e9:.2f} GB/step -> {t:.1f} ms at peak BW; observed itl"
                  f" {peak['itl_ms_p50']} ms -> effective {100 * eff:.0f}% of peak BW")
        else:
            print(f"    if {n_run} running: {byts / 1e9:.2f} GB/step -> {t / eff:.1f} ms predicted itl at the same efficiency")

    # Forecast for the proposed change: cap admission at the batch that fits (max_num_seqs = peak batch).
    cap = peak["batch_size"]
    small = {r["batch_size"]: r for r in sweep}
    print(f"  FORECAST max_num_seqs={cap} (no preemption; extra requests queue). Closed-burst wall = waves of measured rows:")
    fc = []
    for r in after:
        rest = r["num_requests"] - cap
        if rest in small:
            wall = peak["wall_clock_s"] + small[rest]["wall_clock_s"]
        elif rest % cap == 0:
            wall = peak["wall_clock_s"] * (1 + rest // cap)
        else:
            continue
        g = r["num_requests"] * r["gen_len"] / wall
        fc.append({"num_requests": r["num_requests"], "forecast_wall_s": wall, "forecast_gen_tok_s": g,
                   "observed_wall_s": r["wall_clock_s"], "observed_gen_tok_s": gen_goodput(r)})
        print(f"    {r['num_requests']} requests: {wall:.2f} s (observed {r['wall_clock_s']}) -> {g:.1f} gen tok/s"
              f" (observed {gen_goodput(r):.1f}, {100 * (g / gen_goodput(r) - 1):+.1f}%)")
    print(f"    sustained load (>= {cap} in flight): steady state = measured batch-{cap} row = {gen_goodput(peak):.1f}"
          f" gen tok/s per GPU = {peak['num_requests'] / peak['wall_clock_s']:.3f} req/s")
    return {"sweep": out, "peak_batch": peak["batch_size"], "bw_efficiency": eff, "forecast": fc}


def b3(s, rows, batch, prompt):
    print(f"\n== B3  what reported_tok_s counts, and honest generated-token goodput (batch={batch}, prompt_len={prompt})")
    worst = max(abs(r["reported_tok_s"] / ((r["prompt_len"] + r["gen_len"]) * r["num_requests"] / r["wall_clock_s"]) - 1)
                for r in rows)
    print(f"  reported_tok_s == (prompt_len + gen_len) x num_requests / wall_clock_s on all {len(rows)} rows"
          f" (max deviation {100 * worst:.3f}%) -> the counter includes prompt (prefill) tokens")
    r = find(rows, batch, prompt)
    m1 = r["num_requests"] * r["gen_len"] / r["wall_clock_s"]
    frac = r["gen_len"] / (r["prompt_len"] + r["gen_len"])
    m2 = r["reported_tok_s"] * frac
    print(f"  CSV line {r['csv_line']}: {', '.join(f'{k}={r[k]:g}' for k in ('batch_size', 'prompt_len', 'gen_len', 'num_requests', 'wall_clock_s', 'reported_tok_s', 'itl_ms_p50'))}")
    print(f"  way 1 (requests x gen_len / wall):  {r['num_requests']} x {r['gen_len']} / {r['wall_clock_s']} s = {m1:.2f} tok/s")
    print(f"  way 2 (counter x generated share):  {r['reported_tok_s']} x {r['gen_len']}/({r['prompt_len']}+{r['gen_len']})"
          f" = {r['reported_tok_s']} x {frac:.4f} = {m2:.2f} tok/s")
    print(f"  agreement: {100 * abs(m1 - m2) / m1:.3f}% (CSV rounding)")
    ub = r["batch_size"] / (r["itl_ms_p50"] / 1e3)
    print(f"  bound (steady-state decode only): batch / itl_p50 = {r['batch_size']} / {r['itl_ms_p50'] / 1e3:.5f} s"
          f" = {ub:.1f} tok/s; goodput is {100 * m1 / ub:.0f}% of it (prefill + ramp-down take the rest of wall time)")
    print(f"  requests/s = {r['num_requests']} / {r['wall_clock_s']} = {r['num_requests'] / r['wall_clock_s']:.3f}")
    # The two REPORT_v0 claims refer to specific rows; skip them if a different log lacks those rows.
    a, b = find(rows, 16, 512, required=False), find(rows, 16, 3584, required=False)
    if a and b:
        print(f"  REPORT 'longer prompts are better' (batch 16): reported {a['reported_tok_s']} vs {b['reported_tok_s']};"
              f" generated {gen_goodput(a):.1f} vs {gen_goodput(b):.1f} tok/s ({100 * (gen_goodput(b) / gen_goodput(a) - 1):+.1f}% for long)")
    else:
        print("  REPORT 'longer prompts are better': skipped (needs batch-16 rows at prompt 512 and 3584)")
    b48 = find(rows, 48, 3584, required=False)
    if b48:
        print(f"  REPORT 'batch 48 -> ~3200': 1607.4 x 48/24 = {1607.4 * 2:.1f}; the log already has batch 48:"
              f" reported {b48['reported_tok_s']}, generated {gen_goodput(b48):.1f} tok/s")
    else:
        print("  REPORT 'batch 48 -> ~3200': skipped (needs the batch-48, prompt-3584 row)")
    return {"csv_line": r["csv_line"], "method1": m1, "method2": m2, "decode_bound": ub,
            "req_per_s": r["num_requests"] / r["wall_clock_s"], "max_counter_deviation": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(STARTER / "bench/bench_log.csv"))
    ap.add_argument("--seq-len", type=int, default=SPEC["max_model_len"])
    ap.add_argument("--gpu-mem-util", type=float, default=SPEC["gpu_mem_util"])
    ap.add_argument("--kv-bytes", type=float, default=SPEC["kv_bytes"], help="bytes per KV element (2=fp16, 1=fp8)")
    ap.add_argument("--anomaly-prompt", type=int, default=3584)
    ap.add_argument("--goodput-batch", type=int, default=24)
    ap.add_argument("--goodput-prompt", type=int, default=3584)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()
    s = {**SPEC, "gpu_mem_util": args.gpu_mem_util, "kv_bytes": args.kv_bytes}
    rows = load_rows(args.csv)
    kv = b1(s, rows, args.seq_len)
    res = {"spec": s, "B1": kv, "B2": b2(s, rows, args.anomaly_prompt, kv),
           "B3": b3(s, rows, args.goodput_batch, args.goodput_prompt)}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
