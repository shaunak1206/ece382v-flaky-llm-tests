#!/bin/bash
# M1 and M2 on the library study (Progress Report 3): every generated suite of one library, run plain, with the seed
# fixture (M1), and with the isolation fixture (M2), each 100 times shuffled with a new hash seed per run
# (rerun.py --mode both, run i seed 1000+i, so the three versions see the same orders). The suite is copied into the
# library checkout (subjects/<name>/zz_<cond>_<fx>) so M2 imports the library's packages.
# usage: bash scripts/library_fixtures.sh <proj> [runs]
set -u
P=$1; N=${2:-100}
cd "$(dirname "$0")/.."
case $P in tabulate) NAME=python-tabulate;; more_itertools) NAME=more-itertools;; *) NAME=$P;; esac
mkdir -p results/libfix
for C in llm_tests gen_oneshot_sonnet gen_oneshot_haiku gen_tp_sonnet gen_plain_sonnet gen_plain_gpt54mini gen_plain_gpt41mini; do
  [ -d "$C/$P" ] || continue
  for FX in plain seed iso; do
    D="subjects/$NAME/zz_${C}_${FX}"; rm -rf "$D"; mkdir -p "$D"
    cp "$C/$P"/test_*.py "$D"/ 2>/dev/null
    [ -f "$C/$P/conftest.py" ] && cp "$C/$P/conftest.py" "$D"/conftest_orig.py
    case $FX in seed) cp mitigations/seed_conftest.py "$D/conftest.py";; iso) cp mitigations/isolate_conftest.py "$D/conftest.py";; esac
    python scripts/rerun.py --name "$C-$P-$FX" --cwd "$PWD/subjects/$NAME" --target "$PWD/$D" --runs "$N" --mode both \
      --out "results/libfix/${C}__${P}__${FX}.json" -- --continue-on-collection-errors --timeout=15 -p no:cacheprovider 2>&1 \
      | grep -E "DONE|FLAKY|no results" || true
  done
done
