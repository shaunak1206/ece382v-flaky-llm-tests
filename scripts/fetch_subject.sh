#!/bin/bash
# Clone one subject library at its pinned commit (from SUBJECTS.txt) and install it editable.
# usage: bash scripts/fetch_subject.sh <proj>   (proj as used in results: tabulate, more_itertools, ...)
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
case "$1" in
  tabulate) name=python-tabulate ;;
  more_itertools) name=more-itertools ;;
  *) name=$1 ;;
esac
read -r _ repo sha < <(grep -E "^$name " SUBJECTS.txt)
[ -z "$sha" ] && { echo "unknown subject $1"; exit 1; }
mkdir -p subjects
if [ ! -d "subjects/$name/.git" ]; then
  git init -q "subjects/$name"
  git -C "subjects/$name" remote add origin "$repo"
  git -C "subjects/$name" fetch -q --depth 1 origin "$sha"
  git -C "subjects/$name" checkout -q FETCH_HEAD
fi
if [ "$name" = filelock ]; then export SETUPTOOLS_SCM_PRETEND_VERSION=3.16.1; else export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0; fi
$PY -m pip install -q -e "subjects/$name"
echo "installed $name at $sha"
