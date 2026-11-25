#!/usr/bin/env bash
# Runs the baseline twice and requires identical result files.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=/mnt/ai/model-dissect-data/weightedit/runs; mkdir -p "$R"
for i in 1 2; do .venv/bin/python exploratory/weightedit/baseline.py "$R/baseline-run$i.json" > "$R/baseline-run$i.log" 2>&1; done
cmp "$R/baseline-run1.json" "$R/baseline-run2.json" && echo IDENTICAL
grep -E '^(minimind|pythia-14m|gpt2|SELECTION)' "$R/baseline-run1.log"
