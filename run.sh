#!/usr/bin/env bash

set -euo pipefail

# Project root (directory of this script).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROXY_URL=""
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export all_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export ALL_PROXY="$PROXY_URL"

python main.py \
  --data-dir "$ROOT_DIR/data" \
  --output-dir output \
  --schema schemas/example.json