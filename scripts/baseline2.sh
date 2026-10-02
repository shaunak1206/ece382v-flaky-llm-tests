#!/bin/bash
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
python scripts/rerun.py --name human-tinydb --cwd subjects/tinydb --target tests --runs 30 --mode both --out results/human/tinydb.both.json -o addopts= 2>&1 | grep -E "DONE|FLAKY|rc=|no results"
python scripts/rerun.py --name human-humanize --cwd subjects/humanize --target tests --runs 30 --mode both --out results/human/humanize.both.json --ignore=tests/test_benchmarks.py 2>&1 | grep -E "DONE|FLAKY|rc=|no results"
