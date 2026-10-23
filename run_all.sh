#!/usr/bin/env bash
# Reproduce every number in the submission from a clean state.
# Usage: ./run_all.sh            (from your-submission/, expects ../starter_kit/)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi
PY=".venv/bin/python -I"
# Keep model/tokenizer downloads inside the repo dir (excluded from the zip).
export HF_HOME="$PWD/.hf_cache" TIKTOKEN_CACHE_DIR="$PWD/.tiktoken_cache" HF_HUB_DISABLE_TELEMETRY=1
REPO_DIR="$(basename "$PWD")"

echo "== 0. starter kit unchanged?"
(cd .. && sha256sum -c "${REPO_DIR}/STARTER_KIT_SHA256.txt")

echo "== 1. baseline: original fertility.py on the starter samples"
(cd ../starter_kit && "../${REPO_DIR}/.venv/bin/python" -I fertility.py \
   --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2)

echo "== 2. A1 corpus (downloads FLORES-200 tarball if missing, verifies sha256)"
$PY partA/scripts/prepare_corpus.py --download
(cd partA/data/flores200_devtest && python3 -I -c "
import json, hashlib
m = json.load(open('manifest.json'))
for lang, f in m['files'].items():
    assert hashlib.sha256(open(f['path'], 'rb').read()).hexdigest() == f['sha256'], lang
print('corpus sha256 OK for', len(m['files']), 'files')")

echo "== 3. A2 audit experiments"
$PY partA/audit/audit_fertility.py > partA/audit/audit_output.txt
tail -1 partA/audit/audit_output.txt

echo "== 4. A3 corrected analysis"
$PY partA/scripts/token_cost.py | tee partA/results/token_cost_output.txt

echo "== 5. Part B"
python3 -I partB/capacity.py | tee partB/capacity_output.txt

echo "== 6. starter kit still unchanged (hashes, and no __pycache__ written by the audit)"
(cd .. && sha256sum -c --quiet "${REPO_DIR}/STARTER_KIT_SHA256.txt")
test ! -e ../starter_kit/__pycache__ || { echo "starter_kit/__pycache__ was created" >&2; exit 1; }

echo "ALL OK"
