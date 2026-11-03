#!/usr/bin/env python3
"""Build results/gruber/candidates_ext.csv, the replication batch of the matched study (E2).

Gruber et al. (ICST 2021) list every flaky test they found (results/gruber/TestsOverview.csv). For each project in
results/gruber/nod_projects.csv (projects with a flaky test that is not order dependent) that is not already in
candidates.csv (the 58 hand labeled tests), pick one test:
  - Verdict_sameOrder is Flaky and Order-dependent is False,
  - neither the project name nor the test file, class or function name points to the network (NET, PNAME),
  - prefer tests not marked Flaky_Infrastructure, then the failure rate closest to 0.5.
These tests have no hand label, so category is 'unlabeled'.

  python scripts/select_ext.py
"""
import csv, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = f"{ROOT}/results/gruber"
NET = re.compile(r'http|request|url|download|api|client|server|socket|network|connect|online|remote|web|fetch|upload|aws|s3|gcs|ftp|smtp|email|slack|twitter|github|proxy|docker|redis|mongo|postgres|mysql|kafka', re.I)
PNAME = re.compile(r'http|request|api|client|server|socket|network|web|aws|slack|twitter|tweet|proxy|docker|redis|mongo|postgres|mysql|kafka|google|transl|bot|scrap|crawl|spider|telegram|discord|cloud|weather|stock|finance|download', re.I)


def main():
    rows = list(csv.DictReader(open(f"{G}/TestsOverview.csv")))
    cand = {r["Project_Name"] for r in csv.DictReader(open(f"{G}/candidates.csv"))}
    nod = {r["Project_Name"]: r for r in csv.DictReader(open(f"{G}/nod_projects.csv"))}
    by = {}
    for r in rows:
        p = r["Project_Name"]
        if p not in nod or p in cand or PNAME.search(p): continue
        if r["Verdict_sameOrder"] != "Flaky" or r["Order-dependent"] != "False": continue
        if NET.search(" ".join([r["Test_filename"], r["Test_classname"], r["Test_funcname"]])): continue
        n = float(r["#Runs_sameOrder"] or 0)
        if not n: continue
        rate = (float(r["Failed_sameOrder"] or 0) + float(r["Error_sameOrder"] or 0)) / n
        key = (r["Flaky_Infrastructure"] != "True", min(rate, 1 - rate))
        if p not in by or key > by[p][0]: by[p] = (key, r, rate)
    hdr = open(f"{G}/candidates.csv").readline().strip().split(",")
    w = csv.writer(open(f"{G}/candidates_ext.csv", "w", newline="")); w.writerow(hdr)
    for p, (_, r, rate) in sorted(by.items()):
        w.writerow([p, r["Project_URL"], r["Project_Hash"], r["Test_filename"], r["Test_classname"], r["Test_funcname"],
                    r["Test_parametrization"], "unlabeled", rate, nod[p]["tests"]])
    print(len(by), "projects")


if __name__ == "__main__":
    main()
