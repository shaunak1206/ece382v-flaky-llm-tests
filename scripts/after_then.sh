#!/bin/bash
# wait until <done file> has <n> lines, then run a jobs worker on <jobs file>
cd /Users/shaunkapoor/ece382v-flaky-llm-tests
while [ "$(wc -l < "$1" 2>/dev/null || echo 0)" -lt "$2" ]; do sleep 60; done
bash scripts/jobs_worker.sh "$3"
