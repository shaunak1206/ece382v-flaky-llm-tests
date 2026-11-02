#!/bin/bash
# Large rerun campaign for one suite (November 2026): many more runs than the first study, on GitHub Actions.
# usage: bash scripts/scale_reruns.sh <cond> <proj> <runs_both> <runs_tz> <runs_stressng>
#   cond is human, llm_tests, gen_oneshot_sonnet, gen_oneshot_haiku, gen_tp_sonnet, gen_plain_sonnet,
#   gen_plain_gpt54mini or gen_plain_gpt41mini
# Output: results/runs_scale/<cond>/<proj>.<mode>.json in the rerun.py format (see docs/DATA.md).
set -u
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"
cond=$1; proj=$2
declare -a MODES=("both:$3" "tz:$4" "stressng:$5")
mkdir -p "results/runs_scale/$cond"
for mr in "${MODES[@]}"; do
  mode=${mr%%:*}; runs=${mr##*:}
  [ "$runs" = "0" ] && continue
  out="$R/results/runs_scale/$cond/$proj.$mode.json"
  if [ "$cond" = "human" ]; then
    python scripts/mixed.py human "$proj" --mode "$mode" --runs "$runs" --out "$out" 2>&1 | grep -E "DONE|FLAKY|rc=[^01]|no results" || true
  else
    work="$R/results/work/scale/$cond/$proj/$mode"; mkdir -p "$work"
    python scripts/rerun.py --name "$cond-$proj" --cwd "$work" --target "$R/$cond/$proj" --runs "$runs" --mode "$mode" \
      --out "$out" -- --continue-on-collection-errors --timeout=15 2>&1 | grep -E "DONE|FLAKY|rc=[^01]|no results" || true
  fi
done
