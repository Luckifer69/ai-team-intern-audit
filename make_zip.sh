#!/usr/bin/env bash
# Build ../your-submission.zip (top-level dir your-submission/) and verify what went in.
# Usage: ./make_zip.sh [NAME.zip]   (from anywhere; NAME.zip is written next to your-submission/;
#        run ./run_all.sh first if outputs changed)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-your-submission.zip}
case "$OUT" in */*|*[!A-Za-z0-9._-]*|"") echo "zip name must be a plain file name" >&2; exit 2;; esac
case "$OUT" in *.zip) ;; *) echo "zip name must end in .zip" >&2; exit 2;; esac
EXCL=(.venv .hf_cache .tiktoken_cache partA/data/raw)

rm -f "$OUT"
args=()
for d in "${EXCL[@]}"; do args+=("your-submission/$d/*"); done
zip -qr -X "$OUT" your-submission -x "${args[@]}" '*/__pycache__/*' '*.DS_Store' '*/__MACOSX/*'

echo "== verify $OUT"
python3 -I - "$OUT" "${EXCL[@]}" <<'EOF'
import os, sys, zipfile
out, excl = sys.argv[1], sys.argv[2:]
z = zipfile.ZipFile(out)
assert z.testzip() is None, "CRC error"
names = z.namelist()
bad = [n for n in names if not n.startswith("your-submission/")]
bad += [n for n in names if any(n.startswith(f"your-submission/{d}/") for d in excl)
        or "__pycache__/" in n or ".DS_Store" in n or "__MACOSX" in n or n.endswith((".tar.gz", ".pyc"))]
assert not bad, f"unexpected entries: {bad[:10]}"
# every file on disk that should be shipped is in the archive
want = []
for root, dirs, files in os.walk("your-submission"):
    dirs[:] = [d for d in dirs if d != "__pycache__"
               and os.path.relpath(os.path.join(root, d), "your-submission") not in excl]
    want += [os.path.join(root, f) for f in files if f != ".DS_Store"]
missing = sorted(set(want) - set(names))
assert not missing, f"missing from zip: {missing}"
files = [n for n in names if not n.endswith("/")]
print(f"OK: {len(files)} files, {os.path.getsize(out):,} bytes, all under your-submission/, no caches/raw/bytecode")
EOF
