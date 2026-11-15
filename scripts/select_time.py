#!/usr/bin/env python3
"""Build results/gruber/candidates_time.csv, the non randomness batch of the matched study (Progress Report 3).

The matched study is mostly about randomness (35 of 41 projects). This batch looks for more time, concurrency, and
async wait flaky tests in the data of Gruber et al. (results/gruber/TestsOverview.csv), with a rule fixed before any run:
  - Verdict_sameOrder is Flaky and Order-dependent is False (same as select_ext.py),
  - the test file, class, or function name matches TIME (time, clock, sleep, wait, timeout, thread, async, lock, ...),
  - the test is not a lint check run as a test (FLAKE8, PEP8, pylint, mypy, isort, black),
  - neither the project name nor the test names point to the network (NET and PNAME of select_ext.py),
  - the test was not tried before (candidates.csv, candidates_ext.csv), and the project did not fail to install or
    clone in an earlier reproduction (results/gruber/repro*/),
  - one test per project, preferring tests not marked Flaky_Infrastructure, then the failure rate closest to 0.5,
  - at most 40 tests, taken in project name order.

  python scripts/select_time.py
"""
import csv, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_ext import NET, PNAME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = f"{ROOT}/results/gruber"
TIME = re.compile(r"time|clock|sleep|wait|delay|timeout|expir|ttl|duration|schedul|timer|thread|async|await|concurr|"
                  r"parallel|lock|race|queue|worker|pool|future|event", re.I)
LIMIT = 40


def main():
    tried = {(r["Project_Name"], r["Test_funcname"]) for f in ("candidates.csv", "candidates_ext.csv")
             for r in csv.DictReader(open(f"{G}/{f}"))}
    broken = set()
    for f in glob.glob(f"{G}/repro*/*.json"):
        j = json.load(open(f))
        if not j.get("install_ok", True):
            broken.add(j["project"])
    by = {}
    for r in csv.DictReader(open(f"{G}/TestsOverview.csv")):
        p = r["Project_Name"]
        if r["Verdict_sameOrder"] != "Flaky" or r["Order-dependent"] != "False": continue
        if p in broken or PNAME.search(p) or (p, r["Test_funcname"]) in tried: continue
        names = " ".join([r["Test_filename"], r["Test_classname"], r["Test_funcname"]])
        if NET.search(names) or not TIME.search(names): continue
        if re.fullmatch(r"(?i)flake8|pep8|pylint|mypy|isort|black|pycodestyle", r["Test_funcname"]): continue  # lint checks
        n = float(r["#Runs_sameOrder"] or 0)
        if not n: continue
        rate = (float(r["Failed_sameOrder"] or 0) + float(r["Error_sameOrder"] or 0)) / n
        key = (r["Flaky_Infrastructure"] != "True", min(rate, 1 - rate))
        if p not in by or key > by[p][0]:
            by[p] = (key, r, rate)
    hdr = open(f"{G}/candidates.csv").readline().strip().split(",")
    w = csv.writer(open(f"{G}/candidates_time.csv", "w", newline="")); w.writerow(hdr)
    picked = sorted(by.items())[:LIMIT]
    for p, (_, r, rate) in picked:
        w.writerow([p + "@time" if p in {x for x, _ in tried} else p, r["Project_URL"], r["Project_Hash"], r["Test_filename"],
                    r["Test_classname"], r["Test_funcname"], r["Test_parametrization"], "unlabeled", rate, ""])
    print(f"{len(by)} eligible projects, wrote {len(picked)} to candidates_time.csv")


if __name__ == "__main__":
    main()
