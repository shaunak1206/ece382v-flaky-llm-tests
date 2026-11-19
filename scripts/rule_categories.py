#!/usr/bin/env python3
"""Root cause categories assigned by a fixed rule from the diagnosis, for flaky tests without human labels yet.

Rule, in order: (1) project overrides for mechanisms found by hand in the code (kevinarpe-rambutan3: unstable object
hash; docci: time; kindle-maker: other, folder collision); (2) the polluter search verdict: a polluter found ->
test order dependency (randomness if a 10 run rerun did not confirm it), fails alone only with a varied hash seed -> randomness (salted hash), fails alone ->
randomness (unseeded generator); (3) never failed in the search -> test order dependency if it never failed in file
order in its 200 runs, else randomness (unseeded generator).
Validated against Manas's labels of the 78 records of labels.json (where the diagnosis exists), then applied to the
kept flaky held out tests. Writes results/rule_categories.json.

  python scripts/rule_categories.py
"""
import collections, csv, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kappa import norm, kappa

OVERRIDE = {"kevinarpe-rambutan3": "unstable object hash", "docci": "time", "kindle-maker": "other"}


def rule(project, verdict, fails_file_order, confirmed=""):
    if project in OVERRIDE: return OVERRIDE[project]
    if verdict == "od": return "randomness unseeded" if str(confirmed) == "False" else "test order dependency"
    if verdict == "nod-hash": return "randomness salted hash"
    if verdict == "nod": return "randomness unseeded"
    return "test order dependency" if str(fails_file_order) == "0" else "randomness unseeded"


def main():
    diag = {}
    for f in glob.glob(f"{ROOT}/results/od_diagnose*/*.json"):
        j = json.load(open(f))
        for t in j.get("tests", []): diag[(j["project"], t["suite"], t["test"])] = t["verdict"]
    fl = {(r["project"], r["suite"], r["test_id"].split("::")[-1]): r for r in csv.DictReader(open(f"{ROOT}/results/flaky_tests.csv"))}
    pairs = []
    for r in json.load(open(f"{ROOT}/results/root_causes/labels.json")):
        k = (r["project"], r["suite"], r["test"])
        if k not in diag: continue
        sub = r.get("subcategory", "") or ""
        first = norm(r["category"] + " " + sub if "hash" in sub else r["category"])
        pairs.append((first, norm(rule(r["project"], diag[k], (fl.get(k) or {}).get("fails_file_order", "")))))
    kv, agree = kappa(pairs)
    held = collections.defaultdict(collections.Counter)
    for k, r in fl.items():
        if r["arm"] != "held out" or r["kept"] != "True": continue
        held[r["model"]][rule(r["project"], diag.get(k, ""), r["fails_file_order"], r["polluter_confirmed"])] += 1
    out = {"validation": {"records": len(pairs), "kappa": kv, "agreement": agree,
                          "disagreements": collections.Counter(f"{a} -> {b}" for a, b in pairs if a != b)},
           "held_out_kept": {m: dict(c) for m, c in held.items()}}
    json.dump(out, open(f"{ROOT}/results/rule_categories.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
