#!/bin/bash
# usage: llm_rerun.sh <project> [runs]   -- rerun llm_tests/<project> under random order + hash seed
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
n=$1; RUNS=${2:-30}
python scripts/rerun.py --name llm-$n --cwd . --target llm_tests/$n --runs $RUNS --mode both \
    --out results/llm/$n.both.json 2>&1 | grep -E "DONE|FLAKY|rc=|no results"
