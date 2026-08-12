#!/usr/bin/env bash

set -euo pipefail

PROXY_URL=""
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export all_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export ALL_PROXY="$PROXY_URL"

python main.py \
  --data-dir /home/huarui/pythonProject/data_generate/灵枢数据补充/知识图谱智能体/文字识别/output \
  --output-dir output \
  --schema schemas/example.json