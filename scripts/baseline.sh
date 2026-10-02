#!/bin/bash
# Rerun each subject's own (human-written) test suite under random order + hash seed.
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
RUNS=${RUNS:-30}
target() { case $1 in
  cachetools) echo tests;; schedule) echo test_schedule.py;; tinydb) echo tests;;
  humanize) echo tests;; toolz) echo toolz;; python-tabulate) echo test;; esac; }
extra() { case $1 in
  tinydb) echo '-o addopts=';; humanize) echo '--ignore=tests/test_benchmarks.py';; esac; }
for n in cachetools schedule tinydb humanize toolz python-tabulate; do
  python scripts/rerun.py --name human-$n --cwd subjects/$n --target "$(target $n)" --runs $RUNS --mode both \
      --out results/human/$n.both.json $(extra $n) 2>&1 | grep -E "DONE|FLAKY|rc=|no results"
done
echo BASELINE_COMPLETE
