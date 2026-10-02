#!/bin/bash
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
for mode in both tz stress; do
  runs=30; [ $mode != both ] && runs=20
  python scripts/rerun.py --name human-tinydb --cwd subjects/tinydb --target tests --runs $runs --mode $mode --out results/human/tinydb.$mode.json -- -o addopts= 2>&1 | grep -E "DONE|FLAKY|rc=|no results|error"
  python scripts/rerun.py --name human-humanize --cwd subjects/humanize --target tests --runs $runs --mode $mode --out results/human/humanize.$mode.json -- --ignore=tests/test_benchmarks.py 2>&1 | grep -E "DONE|FLAKY|rc=|no results|error"
done
echo HUMANFIX_COMPLETE
