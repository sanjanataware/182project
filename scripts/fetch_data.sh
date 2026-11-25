#!/usr/bin/env bash
# Fetch the public datasets and rebuild the evaluation set used in the paper, byte for byte.
#   * IHEval (CC BY-NC-ND 4.0) is cloned unmodified at the commit we used.
#   * Nemotron-Instruction-Following-Chat-v1 filler turns are streamed by ihc/build_eval_set.py.
# Then every file in data/ is checked against data/SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
IHEVAL_COMMIT=726a62924c3050045954df94347d53fe2bd1090d

if [ ! -d third_party/IHEval ]; then
  git clone https://github.com/ytyz1307zzh/IHEval third_party/IHEval
fi
git -C third_party/IHEval checkout -q "$IHEVAL_COMMIT"
$PY -c "import nltk; nltk.download('punkt_tab', quiet=True)"

$PY ihc/build_eval_set.py
$PY - <<'PYEOF'
import json
rows = [l for l in open("data/eval_test.jsonl") if json.loads(l)["depth"] == 0]
open("data/eval_test_d0.jsonl", "w").writelines(rows)
print(len(rows), "depth-0 rows -> data/eval_test_d0.jsonl")
PYEOF

(cd data && shasum -a 256 -c SHA256SUMS) && echo "data/ matches the paper's eval set"
