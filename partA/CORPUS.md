# A1 — Evaluation corpus

## Source and license [source facts, verified 2026-10-22]

| field | value |
|---|---|
| corpus | FLORES-200, original NLLB release (2022), **devtest** split |
| URL | `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (linked from github.com/facebookresearch/flores, `flores200/README.md`) |
| tarball | 25,585,843 B, HTTP Last-Modified 2022-07-14, sha256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6` |
| retrieved | 2026-10-22 |
| license | **CC-BY-SA 4.0**. The tarball's own README has no license text; I verified the license in the upstream repo README's "Licenses" section ("FLORES-200: CC-BY-SA 4.0"). The repo is archived; newer corrected versions live at FLORES+ (openlanguagedata), which I did not use. |
| citation | NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered Machine Translation*; Goyal et al. (2021), *The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation*. |
| redistribution | The 8 extracted files in `data/flores200_devtest/` are redistributed under CC-BY-SA 4.0 with this attribution, byte-identical to the tarball members (sha256 checked; see `data/flores200_devtest/NOTICE`). The raw tarball is not shipped. |

## Why this corpus

The metric question needs **the same content in every language**. FLORES is professionally
translated and sentence-aligned (line *i* means the same thing in every file), so tokens per
sentence can hold meaning constant. It covers all of English, Hindi, the four major Dravidian
languages and the two other Part C languages (Bengali, Marathi). It is a widely used public
benchmark, so others can reproduce the numbers.

## Contents [measured, from `data/flores200_devtest/manifest.json`]

- **8 languages × 1,012 aligned sentences** (8,096 lines): eng_Latn, hin_Deva, kan_Knda, tam_Taml,
  tel_Telu, mal_Mlym, ben_Beng, mar_Deva. All 1,012 rows are aligned: per-language line counts match
  each other and the 1,012 metadata rows.
- **Domain:** Wikipedia-family articles; wikibooks 351, wikinews 341, wikivoyage 320 sentences
  (`metadata_devtest.tsv`). Encyclopaedic/news/travel register, written, formal.

Per-sentence means, computed on the verbatim lines (min–max shown for words):

| lang | whitespace words | graphemes | UTF-8 bytes | bytes/grapheme | code points/grapheme | file sha256 (prefix) |
|---|---|---|---|---|---|---|
| eng_Latn | 21.64 (7–60) | 130.4 | 130.53 | 1.00 | 1.00 | 612e9fbe8799 |
| hin_Deva | 25.34 (8–67) | 84.96 | 333.1 | 3.92 | 1.52 | 5f5fd39acadc |
| kan_Knda | 15.91 (5–45) | 89.4 | 371.03 | 4.15 | 1.53 | 58e8ed5ef79c |
| tam_Taml | 16.58 (4–42) | 98.54 | 416.64 | 4.23 | 1.55 | a18b26bf278e |
| tel_Telu | 16.74 (5–44) | 75.69 | 349.52 | 4.62 | 1.73 | 2dc641b4fb69 |
| mal_Mlym | 14.75 (5–42) | 78.49 | 406.87 | 5.18 | 1.88 | fe08cf95fc45 |
| ben_Beng | 19.27 (6–52) | 80.72 | 340.59 | 4.22 | 1.58 | 6699aa77b4c9 |
| mar_Deva | 18.82 (6–45) | 79.53 | 351.49 | 4.42 | 1.66 | 63d0d1bfdcfa |

The table already shows the A3 point. For the same meaning, word counts range from 14.75
(Malayalam, agglutinative) to 25.34 (Hindi, analytic). Bytes per grapheme range from 1.00 to 5.18.
"Per word" and "per character" are therefore different amounts of content in different languages.

## Preprocessing and checks

- **Corpus files are verbatim.** Only the file's trailing newline is removed. No filtering,
  normalisation, lowercasing or deduplication. Every text decision lives in the analysis code
  (`scripts/common.py`): strip outer whitespace per line, nothing else. NFC and collapsed whitespace
  are measured as sensitivities in A3.
- **Language/script identification:** for each line, the share of letters (Unicode category L*)
  whose Unicode character name starts with the expected script (LATIN, DEVANAGARI, KANNADA, …). Mean
  share is 97.7% (tel) to 100% (eng).
  - Three Telugu lines fall below 50% (lines 187, 428, 755). On inspection they are Latin-script
    named entities ("Machu Picchu", "Falkland pound (FKP)") inside Telugu sentences: genuine
    code-mixing, so they are kept.
  - FLORES language labels are trusted; I did not run a statistical language-ID model.
- **Sanity results:**
  - 0 empty lines and 0 duplicate lines in every language.
  - Lines with double spaces: kan 213, tel 136, tam 75, ben 71, mar 67, mal 54, hin 6, eng 0.
  - Lines with leading/trailing whitespace: mar 184, kan 83, tel 30, hin 18.
  - Lines not in NFC: ben 609, hin 93, kan 7, mal 4, tam 2, tel 1. The cause is precomposed nukta
    letters, e.g. য় U+09DF, which NFC *decomposes*.
  - These properties matter for A2 (double spaces, NFC) and are measured there.
- **Sampling:** none. The full devtest split is used, so no seed is needed. (The bootstrap in A3 uses
  seed 0.)

## Reproduce

```bash
.venv/bin/python -I partA/scripts/prepare_corpus.py --download   # fetch, verify sha256, rebuild
sha256sum partA/data/flores200_devtest/*.txt                      # compare with manifest.json
```

## What this corpus cannot tell us

- **Size:** 1,012 sentences per language is enough to pin the FLORES number tightly; the bootstrap
  CIs on the token premium are about ±1%. That precision is only about *this* text.
- **Domain and register:** formal Wikipedia-style prose. Real assistant traffic is conversational
  and short, often code-mixed (Hinglish, English technical terms), sometimes romanised (Hindi in
  Latin script, which tokenizes completely differently), and contains numbers, URLs, code and
  markdown. None of that is represented. Part C's "casual" target register is also absent.
- **Translationese:** every Indic line here is a translation of an English source sentence. The
  upstream README notes some FLORES languages were translated from other pivots; I can't verify the
  pivot per language from the tarball. Translations tend to be longer and more formal than native
  text, and they keep English named entities. That likely *overstates* Indic token counts relative
  to native writing, but I have not measured this.
- **Request/response symmetry:** a parallel sentence holds the *prompt* content constant. It does
  not tell us whether the model's *responses* in Indic languages are the same length as in English.
  Decode tokens dominate serving cost.
- **Coverage:** 8 languages, one script each; standard orthography only (no ZWJ/ZWNJ-heavy or
  legacy encodings). Gujarati, Odia and Punjabi are excluded, as are dialects.
- **Version:** this is the 2022 release, not FLORES+, which has later corrections.
