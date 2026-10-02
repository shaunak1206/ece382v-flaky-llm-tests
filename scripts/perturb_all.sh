#!/bin/bash
# Run tz and stress modes over all human + LLM suites (queued after any running reruns finish).
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
RUNS=${RUNS:-20}
while pgrep -f "rerun.py --name (human|llm)-" >/dev/null; do sleep 5; done
target() { case $1 in cachetools) echo tests;; schedule) echo test_schedule.py;; tinydb) echo tests;;
  humanize) echo tests;; toolz) echo toolz;; python-tabulate) echo test;; esac; }
extra() { case $1 in tinydb) echo '-o addopts=';; humanize) echo '--ignore=tests/test_benchmarks.py';; esac; }
llmdir() { case $1 in python-tabulate) echo tabulate;; *) echo $1;; esac; }
for mode in tz stress; do
  for n in cachetools schedule tinydb humanize toolz python-tabulate; do
    python scripts/rerun.py --name human-$n --cwd subjects/$n --target "$(target $n)" --runs $RUNS --mode $mode \
        --out results/human/$n.$mode.json $(extra $n) 2>&1 | grep -E "DONE|FLAKY|rc=|no results"
    python scripts/rerun.py --name llm-$n --cwd . --target llm_tests/$(llmdir $n) --runs $RUNS --mode $mode \
        --out results/llm/$(llmdir $n).$mode.json 2>&1 | grep -E "DONE|FLAKY|rc=|no results"
  done
done
echo PERTURB_COMPLETE
