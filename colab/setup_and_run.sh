#!/bin/bash
# Sets up the project on a fresh Linux machine (Colab, a VM, or a CI runner) and runs the
# extended experiment. Safe to run again after a disconnect: it skips setup that is already done
# and the experiment resumes from the chunks already saved in OUT.
#
# usage: bash setup_and_run.sh <bundle.tar.gz> <work dir> <out dir> [count] [workers]
set -e
BUNDLE=$1; WORK=$2; OUT=$3; COUNT=${4:-100}; WORKERS=${5:-$(nproc 2>/dev/null || echo 2)}
mkdir -p "$WORK" "$OUT"
cd "$WORK"
if [ ! -f ece382v-flaky-llm-tests/scripts/extended.py ]; then tar xzf "$BUNDLE"; fi
cd ece382v-flaky-llm-tests
PY=${PYTHON:-python3}
if [ ! -f .deps_done ]; then
  $PY -m pip install -q -r colab/requirements.txt
  touch .deps_done
fi
mkdir -p subjects
grep -v '^#' SUBJECTS.txt | while read -r name repo sha; do
  [ -z "$name" ] && continue
  if [ ! -d "subjects/$name/.git" ]; then
    git init -q "subjects/$name"
    git -C "subjects/$name" remote add origin "$repo"
    git -C "subjects/$name" fetch -q --depth 1 origin "$sha"
    git -C "subjects/$name" checkout -q FETCH_HEAD
  fi
  if [ ! -f "subjects/$name/.installed" ]; then
    if [ "$name" = filelock ]; then export SETUPTOOLS_SCM_PRETEND_VERSION=3.16.1; else export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0; fi
    $PY -m pip install -q -e "subjects/$name" && touch "subjects/$name/.installed"
    unset SETUPTOOLS_SCM_PRETEND_VERSION
  fi
done
echo "setup done, $(nproc 2>/dev/null) cpus, $($PY --version)"
$PY scripts/extended.py --root "$PWD" --subjects "$PWD/subjects" --out "$OUT" --count "$COUNT" --workers "$WORKERS"
