#!/usr/bin/env python3
"""
audit_fertility.py -- controlled experiments on starter_kit/fertility.py.

The original script is imported read-only from starter_kit/. `analyze_variant()` below is a
re-implementation of fertility.read_lines + fertility.analyze with one switch per suspected
issue. Experiment E0 proves that with all switches at their v0 defaults it reproduces the
original numbers exactly. The experiments are of three kinds:
  - single-factor comparisons: E1, E2, E3, E4, E7 flip exactly one switch; E5 part 1 keeps the
    token counts fixed and changes only the denominator (per word vs per parallel sentence);
  - a combined pipeline comparison: E5 part 2, v0 defaults vs the corrected pipeline (several
    factors at once; it shows the total gap, not its cause);
  - edge-case and equivalence tests: E6 (output diff without random.seed), E8 (injected blank
    lines), E9 (crash inputs), E10 (line alignment of the starter samples).

Usage:
    python -I partA/audit/audit_fertility.py                 # all experiments
    python -I partA/audit/audit_fertility.py --only E1 E7    # a subset
    python -I partA/audit/audit_fertility.py --corpus-dir other/dir --langs eng_Latn,hin_Deva

Writes partA/audit/audit_results.json (all numbers printed below).
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True   # importing starter_kit/fertility.py must not write __pycache__ there
HERE = Path(__file__).resolve().parent
SUB = HERE.parents[1]                       # your-submission/
STARTER = SUB.parent / "starter_kit"
sys.path.insert(0, str(SUB / "partA/scripts"))
import regex                                # noqa: E402
from common import TOKENIZERS, load_encoder  # noqa: E402


def import_original():
    spec = importlib.util.spec_from_file_location("fertility_v0", STARTER / "fertility.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V0 = import_original()
SAMPLES = {"eng": STARTER / "corpus_sample/eng_sample.txt",
           "hin": STARTER / "corpus_sample/hin_sample.txt"}
DEFAULTS = dict(split="space", lower=True, nfc=True, agg="line_mean", char="code_points")


@lru_cache(maxsize=None)
def encoder(name):
    enc = load_encoder(name)
    return lru_cache(maxsize=None)(lambda s: len(enc(s)))   # memoised token count


def read_raw(path, nfc):
    """fertility.read_lines with the NFC step made optional."""
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            out.append(unicodedata.normalize("NFC", line) if nfc else line)
    return out


def char_count(s, kind):
    if kind == "code_points":
        return len(s)
    if kind == "graphemes":
        return len(regex.findall(r"\X", s))
    return len(s.encode("utf-8"))


def analyze_variant(path, tok, split="space", lower=True, nfc=True, agg="line_mean",
                    char="code_points"):
    """Return dict(fertility, tok_per_char, tokens, words, chars, lines). v0 == all defaults."""
    ntok = encoder(tok)
    T, W, C, fert, tpc = [], [], [], [], []
    for line in read_raw(path, nfc):
        if lower:
            line = line.lower()
        t = ntok(line)
        w = len(line.split(" ")) if split == "space" else len(line.split())
        c = char_count(line, char)
        T.append(t), W.append(w), C.append(c)
        fert.append(t / w), tpc.append(t / c)
    n = len(T)
    if agg == "line_mean":
        f, p = sum(fert) / n, sum(tpc) / n
    else:  # pooled: ratio of totals
        f, p = sum(T) / sum(W), sum(T) / sum(C)
    return {"fertility": f, "tok_per_char": p, "tokens": sum(T), "words": sum(W),
            "chars": sum(C), "lines": n}


def pct(new, old):
    return 100.0 * (new - old) / old


def corpus_paths(args):
    paths = {f"sample:{k}": v for k, v in SAMPLES.items()}
    for lang in args.langs.split(","):
        paths[lang] = Path(args.corpus_dir) / f"{lang}.txt"
    return paths


def one_factor(paths, toks, **change):
    """Baseline (v0 defaults) vs exactly one changed setting, for every corpus x tokenizer."""
    rows = []
    for tok in toks:
        for name, path in paths.items():
            if name.startswith("sample:") and tok != "gpt2":
                continue
            base = analyze_variant(path, tok, **DEFAULTS)
            new = analyze_variant(path, tok, **{**DEFAULTS, **change})
            rows.append({"tokenizer": tok, "corpus": name,
                         "fert_v0": base["fertility"], "fert_new": new["fertility"],
                         "fert_delta_pct": pct(new["fertility"], base["fertility"]),
                         "tpc_v0": base["tok_per_char"], "tpc_new": new["tok_per_char"],
                         "tpc_delta_pct": pct(new["tok_per_char"], base["tok_per_char"]),
                         "tokens_v0": base["tokens"], "tokens_new": new["tokens"],
                         "words_v0": base["words"], "words_new": new["words"]})
    return rows


def show(rows, cols):
    print(f"  {'tokenizer':<10}{'corpus':<12}" + "".join(f"{c:>15}" for c in cols))
    for r in rows:
        vals = "".join(f"{r[c]:>15.4f}" if isinstance(r[c], float) else f"{r[c]:>15}" for c in cols)
        print(f"  {r['tokenizer']:<10}{r['corpus']:<12}{vals}")


def lang_of(name):
    return name.split(":")[-1].split("_")[0]


# ------------------------------------------------------------------ experiments
def E0(paths, toks):
    """Harness fidelity: variant(defaults) must equal original fertility.analyze exactly."""
    print("E0  harness reproduces original analyze() bit-for-bit (gpt2, original loader)")
    enc = V0.load_tokenizer("gpt2")
    out = []
    for name, path in paths.items():
        orig = V0.analyze(V0.read_lines(str(path)), enc)
        mine = analyze_variant(path, "gpt2", **DEFAULTS)
        ok = orig == (mine["fertility"], mine["tok_per_char"])
        out.append({"corpus": name, "orig": orig, "variant": [mine["fertility"], mine["tok_per_char"]],
                    "identical": ok})
        print(f"  {name:<12} original=({orig[0]:.6f}, {orig[1]:.6f})  identical={ok}")
    assert all(r["identical"] for r in out), "harness does not reproduce v0 -- stop"
    return out


def E1(paths, toks):
    """split(" ") counts an empty 'word' for every extra space (and misses tabs/NBSP)."""
    print("E1  words = line.split(' ')  ->  line.split()   [only the word denominator changes]")
    probe = "please keep the books  in the cupboard."
    print(f"  minimal: {probe!r}: split(' ') -> {len(probe.split(' '))} words "
          f"{probe.split(' ')}, split() -> {len(probe.split())}")
    rows = one_factor(paths, toks, split="ws")
    show(rows, ["fert_v0", "fert_new", "fert_delta_pct", "words_v0", "words_new"])
    return {"probe": probe, "rows": rows}


def E2(paths, toks):
    """.lower() before encoding changes the token count of English (text != what is served)."""
    print("E2  lower=True -> lower=False   [only the text fed to the tokenizer changes]")
    probes = ["NASA and ISRO announced a joint mission update.",
              "Bengaluru International Airport handled record traffic in March."]
    probe_rows = []
    for tok in toks:
        n = encoder(tok)
        for p in probes:
            probe_rows.append({"tokenizer": tok, "text": p, "orig_case": n(p), "lowered": n(p.lower())})
            print(f"  minimal [{tok}] {p!r}: original case {n(p)} tokens, lowercased {n(p.lower())}")
    rows = one_factor(paths, toks, lower=False)
    show(rows, ["tokens_v0", "tokens_new", "fert_delta_pct", "tpc_delta_pct"])
    return {"probes": probe_rows, "rows": rows}


def E3(paths, toks):
    """Mean of per-line ratios vs ratio of totals (what cost actually sums)."""
    print("E3  agg=line_mean -> agg=pooled (sum tokens / sum units)")
    rows = one_factor(paths, toks, agg="pooled")
    show(rows, ["fert_v0", "fert_new", "fert_delta_pct", "tpc_v0", "tpc_new", "tpc_delta_pct"])
    return {"rows": rows}


def E4(paths, toks):
    """'tok/char' counts code points: a script-encoding artefact, not a unit of content."""
    print("E4  char=code_points -> graphemes / utf8 bytes (tok_per_char only)")
    for w in ["books", "किताबें", "ಪುಸ್ತಕ", "புத்தகம்"]:
        print(f"  minimal {w!r}: code points={len(w)}, graphemes={char_count(w, 'graphemes')}, "
              f"bytes={char_count(w, 'bytes')}")
    rows = {}
    for kind in ["graphemes", "bytes"]:
        rows[kind] = one_factor(paths, toks, char=kind)
        print(f"  -> char={kind}")
        show(rows[kind], ["tpc_v0", "tpc_new", "tpc_delta_pct"])
    # Ratio vs English under each 'char' definition (gpt2, v0 settings otherwise).
    print("  hin/eng ratio of tok-per-<unit> on the samples (gpt2), v0 line-mean aggregation:")
    ratios = {}
    for kind in ["code_points", "graphemes", "bytes"]:
        e = analyze_variant(SAMPLES["eng"], "gpt2", **{**DEFAULTS, "char": kind})["tok_per_char"]
        h = analyze_variant(SAMPLES["hin"], "gpt2", **{**DEFAULTS, "char": kind})["tok_per_char"]
        ratios[kind] = h / e
        print(f"    per {kind:<12} hin/eng = {h / e:.2f}x")
    return {"rows": rows, "sample_hin_over_eng": ratios}


def E5(paths, toks):
    """Conceptual: per-word ratio vs same-content token ratio, from the SAME token counts.

    Part 1 (isolated): one preprocessing configuration, pooled sums, identical token counts T.
    per_word_ratio = (T_L / W_L) / (T_eng / W_eng) and premium = T_L / T_eng, so
    per_word_ratio / premium = W_eng / W_L exactly: the only thing that differs is the denominator.
    Run under the corrected text handling and again under v0 text handling to show it doesn't matter.
    Part 2 (combined, NOT isolated): the full v0 pipeline's headline ratio vs the full corrected
    pipeline's premium -- what the report printed vs what we report; 4 settings + the metric differ.
    The starter samples are excluded: they are not line-parallel (E10).
    """
    print("E5  per-word ratio vs same-content token premium (same token counts; only the denominator differs)")
    langs = [n for n in paths if not n.startswith("sample:")]
    eng = next((n for n in langs if n.startswith("eng")), None)
    if eng is None:
        print("  skipped: needs an eng_* corpus parallel to the others")
        return {}
    configs = {"corrected": dict(split="ws", lower=False, nfc=False, agg="pooled"),
               "v0_text": dict(split="space", lower=True, nfc=True, agg="pooled")}
    iso, comb = [], []
    for tok in toks:
        res = {c: {n: analyze_variant(paths[n], tok, **kw) for n in langs} for c, kw in configs.items()}
        print(f"  [{tok}] {'lang':<6}{'per-word ratio':>15}{'premium':>9}{'distortion':>11}"
              f"{'W_eng/W_L':>10}{'distortion (v0 text)':>22}")
        for n in langs:
            row = {"tokenizer": tok, "corpus": n}
            for c, r in res.items():
                b = r[eng]
                pw = (r[n]["tokens"] / r[n]["words"]) / (b["tokens"] / b["words"])
                pr = r[n]["tokens"] / b["tokens"]
                row.update({f"{c}_per_word_ratio": pw, f"{c}_premium": pr,
                            f"{c}_distortion_pct": 100 * (pw / pr - 1),
                            f"{c}_words_ratio": b["words"] / r[n]["words"]})
            iso.append(row)
            print(f"  {'':<{len(tok) + 3}}{lang_of(n):<6}{row['corrected_per_word_ratio']:>15.3f}"
                  f"{row['corrected_premium']:>9.3f}{row['corrected_distortion_pct']:>+10.1f}%"
                  f"{row['corrected_words_ratio']:>10.3f}{row['v0_text_distortion_pct']:>+21.1f}%")
    print("  Part 2 (COMBINED correction, not isolated): v0 pipeline headline vs corrected premium")
    clean = configs["corrected"]
    for tok in toks:
        v0b, clb = analyze_variant(paths[eng], tok, **DEFAULTS), analyze_variant(paths[eng], tok, **clean)
        for n in langs:
            v0, cl = analyze_variant(paths[n], tok, **DEFAULTS), analyze_variant(paths[n], tok, **clean)
            r = {"tokenizer": tok, "corpus": n, "v0_fertility_ratio": v0["fertility"] / v0b["fertility"],
                 "v0_tpc_ratio": v0["tok_per_char"] / v0b["tok_per_char"],
                 "corrected_premium": cl["tokens"] / clb["tokens"]}
            comb.append(r)
        print(f"  [{tok}] " + "  ".join(f"{lang_of(r['corpus'])} {r['v0_fertility_ratio']:.2f}->{r['corrected_premium']:.2f}"
                                      for r in comb if r["tokenizer"] == tok))
    return {"isolated": iso, "combined": comb}


def E6(paths, toks):
    """random.seed(1337) / import random: never used -> removing it changes nothing."""
    print("E6  delete `import random` + `random.seed(1337)` and diff full script output")
    src = (STARTER / "fertility.py").read_text(encoding="utf-8")
    uses = [l.strip() for l in src.splitlines() if "random" in l]
    print(f"  lines mentioning 'random': {uses}")
    stripped = "\n".join(l for l in src.splitlines()
                         if l.strip() not in ("import random", "random.seed(1337)  # reproducibility"))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fertility_noseed.py"
        p.write_text(stripped + "\n", encoding="utf-8")
        argv = ["--corpus", f"eng={SAMPLES['eng']}", "--corpus", f"hin={SAMPLES['hin']}"]
        a = subprocess.run([sys.executable, "-I", str(STARTER / "fertility.py"), *argv],
                           capture_output=True, text=True, check=True).stdout
        b = subprocess.run([sys.executable, "-I", str(p), *argv],
                           capture_output=True, text=True, check=True).stdout
        a2 = subprocess.run([sys.executable, "-I", str(STARTER / "fertility.py"), *argv],
                            capture_output=True, text=True, check=True).stdout
    print(f"  removed {len(src.splitlines()) - len(stripped.splitlines())} lines; "
          f"output identical: {a == b}; original run twice identical: {a == a2}")
    return {"uses": uses, "identical_without_seed": a == b, "deterministic": a == a2, "stdout": a}


def E7(paths, toks):
    """NFC normalisation: does it change token counts? (it does change code points)"""
    print("E7  nfc=True -> nfc=False   [only the normalisation step changes]")
    for ch in ["ज़", "য়"]:   # ज़ (precomposed), য় (precomposed)
        d = unicodedata.normalize("NFC", ch)
        print(f"  minimal U+{ord(ch):04X} -> NFC {' '.join(f'U+{ord(c):04X}' for c in d)}; tokens " +
              ", ".join(f"{t}: {encoder(t)(ch)}->{encoder(t)(d)}" for t in toks))
    rows = one_factor(paths, toks, nfc=False)
    show(rows, ["tokens_v0", "tokens_new", "fert_delta_pct", "tpc_delta_pct"])
    return {"rows": rows}


def E8(paths, toks):
    """strip() + skipping blank lines: harmless for v0's per-language averages."""
    print("E8  inject blank and whitespace-only lines into the sample files; v0 output must not change")
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for k, src in SAMPLES.items():
            lines = src.read_text(encoding="utf-8").split("\n")
            noisy = ["", "   "] + [x for l in lines for x in (l, "\t ")] + [""]
            p = Path(d) / f"{k}.txt"
            p.write_text("\n".join(noisy), encoding="utf-8")
            enc = V0.load_tokenizer("gpt2")
            a = V0.analyze(V0.read_lines(str(src)), enc)
            b = V0.analyze(V0.read_lines(str(p)), enc)
            out[k] = {"clean": a, "noisy": b, "identical": a == b, "lines_added": len(noisy) - len(lines)}
            print(f"  {k}: +{out[k]['lines_added']} blank/ws lines -> identical={a == b}")
    return out


def E9(paths, toks):
    """Robustness: inputs that crash the original script (no effect on the reported numbers)."""
    print("E9  run the original script on two edge-case inputs")
    out = {}
    with tempfile.TemporaryDirectory() as d:
        cases = {"special_token_text": "The model emitted <|endoftext|> early.\n",
                 "blank_only_file": "\n   \n"}
        for name, text in cases.items():
            p = Path(d) / f"{name}.txt"
            p.write_text(text, encoding="utf-8")
            r = subprocess.run([sys.executable, "-I", str(STARTER / "fertility.py"), "--corpus", f"x={p}"],
                               capture_output=True, text=True)
            err = (r.stderr.strip().splitlines() or [""])
            exc = next((l for l in reversed(err) if "Error" in l), err[-1])
            out[name] = {"input": text, "returncode": r.returncode, "exception": exc}
            print(f"  {name}: {text!r} -> returncode {r.returncode}, {exc[:110]}")
    return out


# (eng line, hin line), 1-based. Pairing by my own reading of the Hindi -- glosses in AUDIT.md.
# (eng line, hin line), 1-based line numbers in corpus_sample/. "close": the Hindi says the same thing.
# "partial": related content, not a translation -- (3, 3) Hindi omits "from a small shop near MG Road";
# (7, 10) English is an instruction ("Please keep the books in the cupboard"), Hindi a statement
# ("the books are kept in the cupboard").
SAMPLE_PAIRS = {"close": [(8, 4), (5, 6), (4, 7)], "partial": [(3, 3), (7, 10)]}


def E10(paths, toks):
    """The starter 'parallel' samples are not line-aligned: compare all lines vs matched pairs."""
    print("E10 sample corpora: same line number vs content-matched pairs (text as written)")
    eng = SAMPLES["eng"].read_text(encoding="utf-8").split("\n")[:10]
    hin = SAMPLES["hin"].read_text(encoding="utf-8").split("\n")[:10]
    same_pos = [(i, i) for i in range(1, 11)]
    close, partial = SAMPLE_PAIRS["close"], SAMPLE_PAIRS["partial"]
    sets = {"all 10 lines": same_pos, f"{len(close)} close pairs": close,
            f"{len(close)} close + {len(partial)} partial": close + partial}
    out = []
    for tok in toks:
        n = encoder(tok)
        for label, pairs in sets.items():
            te = sum(n(eng[e - 1].strip()) for e, _ in pairs)
            th = sum(n(hin[h - 1].strip()) for _, h in pairs)
            out.append({"tokenizer": tok, "set": label, "eng_tokens": te, "hin_tokens": th, "ratio": th / te})
            print(f"  {tok:<8} {label:<20} eng {te:>4}  hin {th:>4}  hin/eng = {th / te:.2f}")
    return {"pairs": SAMPLE_PAIRS, "rows": out}


EXPERIMENTS = {"E0": E0, "E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5, "E6": E6, "E7": E7, "E8": E8,
               "E9": E9, "E10": E10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--corpus-dir", default=str(SUB / "partA/data/flores200_devtest"))
    ap.add_argument("--langs", default="eng_Latn,hin_Deva,kan_Knda,tam_Taml,tel_Telu,mal_Mlym,ben_Beng,mar_Deva")
    ap.add_argument("--tokenizers", default=",".join(TOKENIZERS))
    ap.add_argument("--out", default=str(HERE / "audit_results.json"))
    args = ap.parse_args()
    paths, toks = corpus_paths(args), args.tokenizers.split(",")
    results = {}
    for name in args.only:
        results[name] = EXPERIMENTS[name](paths, toks)
        print()
    if set(args.only) == set(EXPERIMENTS):
        Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
