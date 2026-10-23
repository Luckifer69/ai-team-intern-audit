#!/usr/bin/env python3
"""
prepare_corpus.py -- build the Part A eval corpus from FLORES-200 (original 2022 release).

Downloads (optionally) and verifies the FLORES-200 tarball, extracts one split for the
requested languages, runs sanity checks, and writes:

    <out>/<lang>.txt          one sentence per line, line i is parallel across languages
    <out>/manifest.json       provenance, license, hashes, counts, length stats, checks

Text is copied verbatim (only the trailing newline is removed). No normalization, no
lowercasing, no filtering, no dedup -- the point is to tokenize real text as-is. Sanity
checks only *report*; the script fails if alignment breaks or a line is empty.

Usage:
    python -I partA/scripts/prepare_corpus.py --download            # fetch + build
    python -I partA/scripts/prepare_corpus.py --tarball path/to/flores200_dataset.tar.gz
"""

import argparse
import hashlib
import json
import statistics
import sys
import tarfile
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import regex  # for \X (extended grapheme clusters)

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
DEFAULT_LANGS = ["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml",
                 "tel_Telu", "mal_Mlym", "ben_Beng", "mar_Deva"]
# Unicode character-name prefix expected for letters of each script.
SCRIPT_NAME = {"Latn": "LATIN", "Deva": "DEVANAGARI", "Knda": "KANNADA", "Taml": "TAMIL",
               "Telu": "TELUGU", "Mlym": "MALAYALAM", "Beng": "BENGALI"}

ROOT = Path(__file__).resolve().parents[1]  # partA/


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_share(line, script):
    """Fraction of letters (category L*) whose Unicode name starts with the expected script."""
    want = SCRIPT_NAME[script]
    letters = [c for c in line if unicodedata.category(c).startswith("L")]
    if not letters:
        return 1.0
    return sum(unicodedata.name(c, "").startswith(want) for c in letters) / len(letters)


def summary(xs):
    xs = sorted(xs)
    return {"mean": round(statistics.fmean(xs), 2), "median": statistics.median(xs),
            "p5": xs[int(0.05 * (len(xs) - 1))], "p95": xs[int(0.95 * (len(xs) - 1))],
            "min": xs[0], "max": xs[-1], "total": sum(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", default=str(ROOT / "data/raw/flores200_dataset.tar.gz"))
    ap.add_argument("--download", action="store_true", help="download the tarball if missing")
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS))
    ap.add_argument("--out", default=str(ROOT / "data/flores200_devtest"))
    args = ap.parse_args()

    tarball = Path(args.tarball)
    if not tarball.exists():
        if not args.download:
            sys.exit(f"missing {tarball}; rerun with --download or pass --tarball")
        tarball.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} -> {tarball}")
        urllib.request.urlretrieve(URL, tarball)
    digest = sha256_file(tarball)
    if digest != SHA256:
        sys.exit(f"sha256 mismatch for {tarball}: {digest} (expected {SHA256})")

    langs = args.langs.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    texts, metadata = {}, None
    with tarfile.open(tarball) as tar:
        meta_raw = tar.extractfile(f"./flores200_dataset/metadata_{args.split}.tsv").read()
        meta_rows = meta_raw.decode("utf-8").splitlines()[1:]
        metadata = [r.split("\t") for r in meta_rows]
        for lang in langs:
            member = f"./flores200_dataset/{args.split}/{lang}.{args.split}"
            raw = tar.extractfile(member).read().decode("utf-8")
            # Split on "\n" only: str.splitlines() would also split on U+2028 etc. inside a sentence.
            lines = raw.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            texts[lang] = lines

    # ---- sanity checks -------------------------------------------------------------
    n = {lang: len(v) for lang, v in texts.items()}
    if len(set(n.values())) != 1 or len(metadata) != next(iter(n.values())):
        sys.exit(f"alignment failure: line counts {n}, metadata rows {len(metadata)}")
    checks = {}
    for lang, lines in texts.items():
        script = lang.split("_")[1]
        shares = [script_share(l, script) for l in lines]
        empty = [i for i, l in enumerate(lines) if not l.strip()]
        if empty:
            sys.exit(f"{lang}: empty lines at {empty[:10]}")
        dup = sum(c - 1 for c in Counter(lines).values() if c > 1)
        checks[lang] = {
            "empty_lines": 0,
            "duplicate_lines": dup,
            "lines_not_NFC": sum(unicodedata.normalize("NFC", l) != l for l in lines),
            "lines_with_double_space": sum("  " in l for l in lines),
            "lines_with_leading_or_trailing_ws": sum(l != l.strip() for l in lines),
            "lines_with_non_space_whitespace": sum(any(c.isspace() and c != " " for c in l) for l in lines),
            "expected_script_letter_share_mean": round(statistics.fmean(shares), 4),
            "lines_with_expected_script_share_below_0.5": sum(s < 0.5 for s in shares),
        }

    stats, files = {}, {}
    for lang, lines in texts.items():
        path = out / f"{lang}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files[lang] = {"path": path.name, "sha256": sha256_file(path), "lines": len(lines)}
        stats[lang] = {
            "whitespace_words": summary([len(l.split()) for l in lines]),
            "code_points": summary([len(l) for l in lines]),
            "graphemes": summary([len(regex.findall(r"\X", l)) for l in lines]),
            "utf8_bytes": summary([len(l.encode("utf-8")) for l in lines]),
        }

    manifest = {
        "source": {
            "name": "FLORES-200 (original NLLB release, 2022)",
            "url": URL,
            "tarball_sha256": digest,
            "split": args.split,
            "license": "CC-BY-SA 4.0 (per github.com/facebookresearch/flores README, 'Licenses')",
            "citation": "NLLB Team et al. (2022), No Language Left Behind: Scaling Human-Centered "
                        "Machine Translation; Goyal et al. (2021), The FLORES-101 Evaluation Benchmark.",
        },
        "transformations": "verbatim copy of each line; trailing newline removed; nothing else",
        "aligned_sentences": next(iter(n.values())),
        "domains": dict(Counter(r[1] for r in metadata)),
        "files": files,
        "checks": checks,
        "length_stats_per_sentence": stats,
        "tool_versions": {"python": sys.version.split()[0],
                          "unicodedata": unicodedata.unidata_version,
                          "regex": regex.__version__},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")

    print(f"{'lang':<10}{'lines':>6}{'words':>8}{'graphemes':>11}{'bytes':>9}{'dups':>6}"
          f"{'notNFC':>8}{'script%':>9}")
    for lang in langs:
        s, c = stats[lang], checks[lang]
        print(f"{lang:<10}{n[lang]:>6}{s['whitespace_words']['total']:>8}{s['graphemes']['total']:>11}"
              f"{s['utf8_bytes']['total']:>9}{c['duplicate_lines']:>6}{c['lines_not_NFC']:>8}"
              f"{100 * c['expected_script_letter_share_mean']:>8.1f}%")
    print(f"wrote {out}/manifest.json")


if __name__ == "__main__":
    main()
