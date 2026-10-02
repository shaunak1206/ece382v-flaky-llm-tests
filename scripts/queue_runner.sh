#!/bin/bash
# Processes "cond proj" lines from results/queue.txt one at a time, forever until a line "STOP" appears.
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
touch results/queue.txt results/queue.done
while true; do
  line=$(comm -23 <(sort -u results/queue.txt) <(sort -u results/queue.done) | head -1)
  if [ -z "$line" ]; then sleep 10; continue; fi
  [ "$line" = "STOP" ] && { echo "QUEUE_STOPPED"; break; }
  set -- $line; cond=$1; proj=$2
  echo "=== $(date +%H:%M) start $cond/$proj"
  if [ "$cond" != "llm_tests" ]; then python scripts/filter_oneshot.py $cond $proj; fi
  bash scripts/llm_suite_reruns.sh $cond $proj
  echo "$line" >> results/queue.done
  echo "=== $(date +%H:%M) done $cond/$proj"
done
