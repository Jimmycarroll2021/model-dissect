#!/usr/bin/env bash
# Held-out evaluation, twice; each repeat runs the three sets of weights side by side. Result files must be identical.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=/mnt/ai/model-dissect-data/weightedit/runs; RUN=$R/qwen-p25
for i in 1 2; do
  pids=()
  for w in unedited edited random; do
    .venv/bin/python exploratory/weightedit/qwen/qevaluate.py gen "$RUN" $w "$R/qwen-heldout-$w-run$i.json" > "$R/qwen-heldout-$w-run$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  .venv/bin/python exploratory/weightedit/qwen/qevaluate.py verdict "$R"/qwen-heldout-{unedited,edited,random}-run$i.json "$R/qwen-heldout-run$i.json" > "$R/qwen-heldout-run$i.log"
done
for w in unedited edited random; do cmp "$R/qwen-heldout-$w-run1.json" "$R/qwen-heldout-$w-run2.json"; done
cmp "$R/qwen-heldout-run1.json" "$R/qwen-heldout-run2.json" && echo IDENTICAL
cat "$R/qwen-heldout-run1.log"
