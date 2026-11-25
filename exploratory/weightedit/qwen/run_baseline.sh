#!/usr/bin/env bash
# Runs the baseline twice (side by side) and requires identical result files.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=/mnt/ai/model-dissect-data/weightedit/runs; mkdir -p "$R"
pids=()
for i in 1 2; do .venv/bin/python exploratory/weightedit/qwen/qbaseline.py "$R/qwen-baseline-run$i.json" > "$R/qwen-baseline-run$i.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
cmp "$R/qwen-baseline-run1.json" "$R/qwen-baseline-run2.json" && echo IDENTICAL
grep -h '^RESULT' "$R/qwen-baseline-run1.log"
