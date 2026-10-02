#!/bin/bash
# Human-written suites of the 6 expansion subjects, all three modes.
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
for mode in both tz stress; do
  runs=30; [ $mode != both ] && runs=20
  for n in tenacity filelock boltons more-itertools arrow pybreaker; do
    pp=""; [ $n = filelock ] && pp="PYTHONPATH=."
    env $pp python scripts/rerun.py --name human-$n --cwd subjects/$n --target tests --runs $runs --mode $mode \
       --out results/human/$n.$mode.json -- -o addopts= --timeout=120 2>&1 | grep -E "DONE|FLAKY|rc=[^01]|no results|rror:"
  done
done
echo HUMANNEW_COMPLETE
