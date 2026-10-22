#!/bin/bash
# usage: llm_suite_reruns.sh <cond-dir> <proj>   (cond-dir: llm_tests | gen_oneshot_sonnet | gen_oneshot_haiku)
# Reruns the full generated suite (accepted and rejected tests) in an isolated work dir, 3 modes.
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"; [ -f .venv/bin/activate ] && . .venv/bin/activate
cond=$1; proj=$2
for mode in both tz stress; do
  runs=30; [ $mode != both ] && runs=20
  work=$R/results/work/$cond/$proj/$mode; mkdir -p $work
  python scripts/rerun.py --name $cond-$proj --cwd $work --target $R/$cond/$proj --runs $runs --mode $mode \
     --out $R/results/runs/$cond/$proj.$mode.json -- --continue-on-collection-errors --timeout=15 2>&1 \
     | grep -E "DONE|FLAKY|rc=[^01]|no results|rror:"
done
