#!/bin/bash
# Claim-based queue worker; run several in parallel. A suite is claimed with mkdir (atomic).
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
mkdir -p results/claims; touch results/queue.done
while true; do
  picked=""
  while read -r cond proj; do
    [ -z "$cond" ] && continue
    if mkdir "results/claims/${cond}__${proj}" 2>/dev/null; then picked="$cond $proj"; break; fi
  done < <(sort -u results/queue.txt)
  if [ -z "$picked" ]; then echo "WORKER_IDLE $$ $(date +%H:%M)"; break; fi
  set -- $picked; cond=$1; proj=$2
  echo "=== $(date +%H:%M) [w$$] start $cond/$proj"
  [ "$cond" != "llm_tests" ] && python scripts/filter_oneshot.py $cond $proj
  bash scripts/llm_suite_reruns.sh $cond $proj
  echo "$cond $proj" >> results/queue.done
  echo "=== $(date +%H:%M) [w$$] done $cond/$proj"
done
