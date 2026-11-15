#!/usr/bin/env python3
"""Build results/gruber/candidates_od.csv, the order dependent arm (Progress Report 3, step 3).

Gruber et al. mark a flaky test as order dependent when it passes in the same order and flips across random orders.
Rule, fixed before any run:
  - Order-dependent is True and Verdict_randomOrder is Flaky,
  - neither the project name nor the test names point to the network (NET and PNAME of select_ext.py),
  - the project is not in the matched study and did not fail to install or clone in an earlier reproduction,
  - one test per project, preferring tests not marked Flaky_Infrastructure, then the random order failure rate
    closest to 0.5,
  - at most 45 tests, taken in project name order (about 45 tries for 10 projects at our 22% yield).
They are reproduced with scripts/gruber_reproduce.py --shuffle-file (the whole test file in a new random order per run).

  python scripts/select_od.py
"""
import csv, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_ext import NET, PNAME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = f"{ROOT}/results/gruber"
LIMIT = 45


def main():
    used = set(open(f"{G}/matched_projects.txt").read().split())
    broken = {json.load(open(f)).get("project") for f in glob.glob(f"{G}/repro*/*.json") if not json.load(open(f)).get("install_ok", True)}
    by = {}
    for r in csv.DictReader(open(f"{G}/TestsOverview.csv")):
        p = r["Project_Name"]
        if r["Order-dependent"] != "True" or r["Verdict_randomOrder"] != "Flaky": continue
        if p in used or p in broken or PNAME.search(p): continue
        if NET.search(" ".join([r["Test_filename"], r["Test_classname"], r["Test_funcname"]])): continue
        if re.fullmatch(r"(?i)flake8|pep8|pylint|mypy|isort|black|pycodestyle", r["Test_funcname"]): continue
        n = float(r["#Runs_randomOrder"] or 0)
        if not n: continue
        rate = (float(r["Failed_randomOrder"] or 0) + float(r["Error_randomOrder"] or 0)) / n
        key = (r["Flaky_Infrastructure"] != "True", min(rate, 1 - rate))
        if p not in by or key > by[p][0]:
            by[p] = (key, r, rate)
    hdr = open(f"{G}/candidates.csv").readline().strip().split(",")
    w = csv.writer(open(f"{G}/candidates_od.csv", "w", newline="")); w.writerow(hdr)
    picked = sorted(by.items())[:LIMIT]
    for p, (_, r, rate) in picked:
        w.writerow([p, r["Project_URL"], r["Project_Hash"], r["Test_filename"], r["Test_classname"], r["Test_funcname"],
                    r["Test_parametrization"], "order dependent", rate, ""])
    print(f"{len(by)} eligible projects, wrote {len(picked)} to candidates_od.csv")


if __name__ == "__main__":
    main()
