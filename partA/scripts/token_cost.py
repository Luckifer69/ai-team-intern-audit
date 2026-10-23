#!/usr/bin/env python3
"""
token_cost.py -- corrected cross-language tokenizer comparison (Part A3).

For every tokenizer x language it counts, per sentence: tokens (no special tokens, no
lowercasing, no normalisation beyond stripping outer whitespace), whitespace words, extended
grapheme clusters, UTF-8 bytes and code points. Then it reports:

  * pooled tokens per word / grapheme / byte          (ratio of totals, language-internal)
  * token premium vs the base language on the SAME parallel sentences
        premium(L) = sum_i tokens_L[i] / sum_i tokens_base[i]
    with a paired bootstrap 95% CI and the sentence-level median / IQR of tokens_L[i]/tokens_base[i]
  * two sensitivities for the premium: NFC-normalised input, internal whitespace collapsed

Usage:
    python -I partA/scripts/token_cost.py                                  # FLORES, 3 tokenizers
    python -I partA/scripts/token_cost.py --tokenizers gpt2,hf:xlm-roberta-base
    python -I partA/scripts/token_cost.py --corpus eng=a.txt --corpus hin=b.txt --out-dir /tmp/x

If the corpora do not have equal line counts the premium is not computed (not parallel).
"""

import argparse
import csv
import json
import random
import statistics
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TOKENIZERS, load_encoder, read_corpus, units  # noqa: E402

PARTA = Path(__file__).resolve().parents[1]
DEFAULT_LANGS = "eng_Latn,hin_Deva,kan_Knda,tam_Taml,tel_Telu,mal_Mlym,ben_Beng,mar_Deva"
BOOT_REPS, BOOT_SEED = 1000, 0


def quantile(xs, q):
    xs = sorted(xs)
    return xs[round(q * (len(xs) - 1))]


def bootstrap_ratio(num, den, reps=BOOT_REPS, seed=BOOT_SEED):
    """Paired bootstrap over sentence indices for sum(num)/sum(den). Returns (lo, hi) 95%."""
    rng = random.Random(seed)
    n = len(num)
    stats = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(sum(num[i] for i in idx) / sum(den[i] for i in idx))
    return quantile(stats, 0.025), quantile(stats, 0.975)


def count_rows(tok_name, encode, lang, lines):
    rows = []
    for i, text in enumerate(lines):
        u = units(text)
        if min(u.values()) == 0:
            sys.exit(f"{lang} line {i + 1}: empty sentence -- every denominator must be nonzero")
        rows.append({
            "tokenizer": tok_name, "lang": lang, "idx": i, **u,
            "tokens": len(encode(text)),
            "tokens_nfc": len(encode(unicodedata.normalize("NFC", text))),
            "tokens_ws_collapsed": len(encode(" ".join(text.split()))),
        })
    return rows


def summarise(rows_by_lang, base):
    out = []
    parallel = len({len(r) for r in rows_by_lang.values()}) == 1 and base in rows_by_lang
    for lang, rows in rows_by_lang.items():
        tot = {k: sum(r[k] for r in rows) for k in
               ("tokens", "tokens_nfc", "tokens_ws_collapsed", "words", "graphemes", "bytes", "code_points")}
        s = {"tokenizer": rows[0]["tokenizer"], "lang": lang, "sentences": len(rows), **tot,
             "tok_per_word": tot["tokens"] / tot["words"],
             "tok_per_grapheme": tot["tokens"] / tot["graphemes"],
             "tok_per_byte": tot["tokens"] / tot["bytes"],
             "tok_per_sentence": tot["tokens"] / len(rows),
             "tok_per_sentence_median": statistics.median(r["tokens"] for r in rows)}
        if parallel:
            b = rows_by_lang[base]
            num, den = [r["tokens"] for r in rows], [r["tokens"] for r in b]
            per_sent = [x / y for x, y in zip(num, den)]
            lo, hi = bootstrap_ratio(num, den)
            btot = {k: sum(r[k] for r in b) for k in ("tokens", "tokens_nfc", "tokens_ws_collapsed", "words")}
            s.update({
                "premium_vs_base": tot["tokens"] / btot["tokens"],
                "premium_ci95_lo": lo, "premium_ci95_hi": hi,
                "premium_sentence_median": statistics.median(per_sent),
                "premium_sentence_p25": quantile(per_sent, 0.25),
                "premium_sentence_p75": quantile(per_sent, 0.75),
                "premium_nfc": tot["tokens_nfc"] / btot["tokens_nfc"],
                "premium_ws_collapsed": tot["tokens_ws_collapsed"] / btot["tokens_ws_collapsed"],
                "per_word_ratio_vs_base": s["tok_per_word"] / (btot["tokens"] / btot["words"]),
            })
        out.append(s)
    return out, parallel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=str(PARTA / "data/flores200_devtest"))
    ap.add_argument("--langs", default=DEFAULT_LANGS)
    ap.add_argument("--corpus", action="append", metavar="LANG=PATH",
                    help="explicit corpora (repeatable); overrides --corpus-dir/--langs")
    ap.add_argument("--base", default=None, help="base language for the premium (default: first)")
    ap.add_argument("--tokenizers", default=",".join(TOKENIZERS),
                    help="comma list of names in common.TOKENIZERS or tiktoken:<enc> / hf:<repo>[@rev]")
    ap.add_argument("--ref-tokenizer", default="o200k",
                    help="index_vs_ref = tok/sentence / base-language tok/sentence on THIS tokenizer "
                         "(stand-in for the production tokenizer; comparable across tokenizers)")
    ap.add_argument("--out-dir", default=str(PARTA / "results"))
    args = ap.parse_args()

    if args.corpus:
        paths = dict(spec.split("=", 1) for spec in args.corpus)
    else:
        paths = {l: Path(args.corpus_dir) / f"{l}.txt" for l in args.langs.split(",")}
    corpora = {lang: read_corpus(p) for lang, p in paths.items()}
    base = args.base or next(iter(corpora))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], []
    for tok in args.tokenizers.split(","):
        encode = load_encoder(tok)
        rows_by_lang = {lang: count_rows(tok, encode, lang, lines) for lang, lines in corpora.items()}
        s, parallel = summarise(rows_by_lang, base)
        summary += s
        all_rows += [r for rows in rows_by_lang.values() for r in rows]
        if not parallel:
            print(f"WARNING: corpora are not parallel (line counts differ) -- premium not computed")

    ref = [s for s in summary if s["tokenizer"] == args.ref_tokenizer and s["lang"] == base]
    for s in summary:
        if ref and len({x["sentences"] for x in summary}) == 1:
            s["index_vs_ref"] = s["tok_per_sentence"] / ref[0]["tok_per_sentence"]

    with open(out / "per_sentence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(), w.writerows(all_rows)
    fields = list(dict.fromkeys(k for s in summary for k in s))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(), w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                                     for s in summary)
    meta = {"tokenizers": {t: TOKENIZERS.get(t, t) for t in args.tokenizers.split(",")},
            "corpora": {k: str(v) for k, v in paths.items()}, "base": base,
            "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED}, "ref_tokenizer": args.ref_tokenizer,
            "text_processing": "outer whitespace stripped; no lowercasing; no NFC (NFC is a sensitivity column); "
                               "no special tokens"}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    has_p = "premium_vs_base" in summary[0]
    tw = max(10, *(len(s["tokenizer"]) + 1 for s in summary))
    print(f"{'tokenizer':<{tw}}{'lang':<10}{'tok/sent':>9}{'tok/word':>9}{'tok/graph':>10}{'tok/byte':>9}" +
          (f"{'PREMIUM':>9}{'95% CI':>15}{'sent med [IQR]':>20}{'perword-ratio':>14}{'NFC':>7}{'ws':>7}"
           if has_p else "") + (f"{'idx@ref':>9}" if "index_vs_ref" in summary[0] else ""))
    for s in summary:
        line = (f"{s['tokenizer']:<{tw}}{s['lang']:<10}{s['tok_per_sentence']:>9.1f}{s['tok_per_word']:>9.2f}"
                f"{s['tok_per_grapheme']:>10.3f}{s['tok_per_byte']:>9.3f}")
        if has_p:
            line += (f"{s['premium_vs_base']:>9.2f}  [{s['premium_ci95_lo']:.2f}, {s['premium_ci95_hi']:.2f}]"
                     f"   {s['premium_sentence_median']:.2f} [{s['premium_sentence_p25']:.2f}-"
                     f"{s['premium_sentence_p75']:.2f}]{s['per_word_ratio_vs_base']:>14.2f}"
                     f"{s['premium_nfc']:>7.2f}{s['premium_ws_collapsed']:>7.2f}")
        if "index_vs_ref" in s:
            line += f"{s['index_vs_ref']:>9.2f}"
        print(line)
    print(f"\nbase={base}; ref tokenizer for idx@ref={args.ref_tokenizer if ref else 'n/a'}; wrote {out}/summary.csv, per_sentence.csv, run_meta.json")


if __name__ == "__main__":
    main()
