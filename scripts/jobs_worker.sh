#!/bin/bash
# Generic claim-based worker. Each line of the job file is "<job id> <shell command>".
# A job is claimed with mkdir (atomic), so several workers can share one file.
# usage: bash scripts/jobs_worker.sh results/jobs_oct22.txt
cd /Users/shaunkapoor/ece382v-flaky-llm-tests && . .venv/bin/activate
JOBS=$1; CL=results/claims_$(basename "$JOBS" .txt); mkdir -p "$CL"
while true; do
  picked=""
  while read -r id cmd; do
    [ -z "$id" ] && continue
    if mkdir "$CL/$id" 2>/dev/null; then picked="$id"; pcmd="$cmd"; break; fi
  done < "$JOBS"
  [ -z "$picked" ] && { echo "WORKER_IDLE $$ $(date +%H:%M)"; break; }
  echo "=== $(date +%H:%M) [w$$] start $picked"
  bash -c "$pcmd"
  echo "$picked" >> "$CL/DONE"
  echo "=== $(date +%H:%M) [w$$] done $picked"
done
