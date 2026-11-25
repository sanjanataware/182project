#!/usr/bin/env bash
# Download the raw gpt-oss-20b responses (reasoning + final answer for every prompt, seed and
# checkpoint) from the GitHub Release and check them. Only needed to re-score (make score).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${REPO:-sanjanataware/182project}
TAG=${TAG:-v1.0}
ASSET=gpt-oss-20b-responses.tar.gz
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --pattern "$ASSET.sha256" --clobber
else
  curl -fL -o "$ASSET" "$URL" && curl -fL -o "$ASSET.sha256" "$URL.sha256"
fi
shasum -a 256 -c "$ASSET.sha256"
tar -xzf "$ASSET"           # -> responses/gpt-oss-20b__<precision>__s<seed>.jsonl
rm "$ASSET" "$ASSET.sha256"
ls responses | wc -l | xargs echo "response files:"
