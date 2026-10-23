## Memo: casual-register replies (hi, kn, ta, te, bn, mr)

**Recommendation:** launch **(c) prompt engineering**, gated on hi/kn native evidence. **(a) LoRA
SFT on hi/kn only** is the Day-8 fallback. **Defer (b):** no ≤1B multilingual rewriter is
validated, and its latency, memory and second-model cost (below) are hard to justify now; reconsider
if one passes the same gates. Casual only where a native speaker validated it. **Dates:** Day 1 =
Thu 2026-10-22 (writing date; none was given); A100 Day 1–14; launch review Day 22 (Thu 11-12).

**Assumptions** (not measured): A1 own ~4B open model (like FLM-4B), prompt + LoRA access · A2 ~40
pairwise judgments/h · A3 real user prompts per language · A4 A100 ≈312 bf16 TFLOPS at 35%, ~1k gen
tok/s · **S1 serving = Part B's L4** (Part C names none) · A5 ta/te/bn/mr: volunteers only, capacity unknown.

**Back-of-envelope**
- **Paid reviewer (hi + kn only):** 10 h × 3 wk × 40/h = **1,200 judgments**.
  - Prompt path: day-one 120 + iteration 160 + final 300 (100 hi + 100 kn + 100 safety) = **580 (48%)**.
  - If killed on Day 8: 280 used, then SFT QA (10% of 3,000 hi/kn pairs) 300 + re-eval 300 = **880 (73%)**.
  - 6-language (a): QA 10% of 9,000 = 900; only hi/kn's 300 reviewable, **600 not**.
  - Volunteer gate: ≥50 judgments × 4 languages ≈ 5 h; none by Day 15 → formal.
- **A100, hi/kn SFT:** generate 1.8M tokens (3,000 × 2 × 300) ≈ 0.5 h; train 3 epochs × 1.8M ×
  6 × 4.2e9 ≈ 1.4e17 FLOP ≈ 0.35 h; + ~2 h setup/eval [A] ≈ **3 of 336 GPU-h**.
- **Serving on L4 [S1]:** (c) adds 250 × 1.6–2.0 = 400–500 prompt tokens (cached once, else 57 MB KV/seq).
  (b): a 300-token reply at 43.5 ms/token (Part B batch-1 ITL) = 13.0 s. A ≤1B rewriter (2 GB;
    ≥10 ms/token = 2 GB ÷ 300 GB/s ÷ 66%) adds **+3.0 s → ≈16 s (+23%)** and loses streaming.
    Its weights cut KV 12.08 → 10.08 GB (**25 → 21** 4k seqs).

**Trade-offs** (ceiling · serving cost · rollback · risk): **(c)** base model · +0.4–0.5k tokens ·
instant · shift too weak. **(a)** highest · none · adapter swap · synthetic data verifiable
only in hi/kn. **(b)** ≤1B, unvalidated · +3 s, no streaming, −16% KV · two models · meaning drift.

**Success metric (proposed).** 100 held-out real prompts each (hi, kn), blind native pairwise vs
current: **win rate ≥ 70%** (ties ½; CI lower bound ≈ 61%), **meaning changed ≤ 5%**, **0 new
failures** on a 50-prompt safety set per language.

**Decision rules** (hi and kn each). **Launch** only if the full success metric is met. **Iterate,
no launch:** best arm 60–70% with ≤ 10% meaning changes. **Kill criterion, Thu 2026-10-29 (Day 8):**
best arm **< 60%** or **> 10% meaning changes** → hi/kn LoRA SFT, which must pass the same metric.
Nothing passes by the launch review (SFT: by Day 14, Wed 11-04, last A100 day) → stay formal.

**Day-one experiment.** 30 hi + 30 kn real prompts; reviewer writes 3 casual exemplars per
language (30 min). Arms: A0 current, A1 casual instruction, A2 = A1 + exemplars; blind pairwise vs
A0 + meaning flags (120 judgments ≈ 3 h). Pilot, not a launch signal: A2 ≥ 70% → freeze prompt,
recruit ta/te/bn/mr volunteers (hi/kn results don't transfer); A2 < 50% → start SFT data too.
