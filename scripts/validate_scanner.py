#!/usr/bin/env python3
"""Validate the static risk scanner on flaky tests that someone else labeled.

Ground truth is the manual root cause sample of Gruber et al. (ICST 2021, Zenodo 10.5281/zenodo.4450435,
NOD_manual_classification.csv): 100 non order dependent flaky Python tests from 100 PyPI projects with a
hand labeled cause. For each one we check out the test file at the recorded commit and scan it.

Controls are the other test functions in the same files that Gruber et al. did not report as flaky
(neither in the NOD sample nor in victims_brittles.csv). They were not observed to flip in 200 runs,
which does not prove they are stable, so the control flag rate is an upper bound on the false positive rate.

The scanner was written for our twelve subjects before this check. NETWORK is an extra rule added on
October 22 after this check showed network is the largest category in Python; it is reported separately.
"""
import csv, json, os, re, subprocess, sys, collections
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
import smells
V = f"{ROOT}/results/scanner_validation"
REPOS = f"{V}/repos"
NETWORK = re.compile(r"\brequests\.(get|post|put|delete|head|Session)\(|urllib\.request|urlopen\(|http\.client|"
                     r"\bsocket\.(socket|create_connection)\(|aiohttp\.|httpx\.|smtplib\.|ftplib\.|\bconnect\(\s*\(?['\"]")

# which Luo et al. / Gruber et al. root cause category each rule is meant to catch
RULE_SOURCES = {
 "SLEEP": "async wait, time (Luo et al. 2014; Gruber et al. 2021)",
 "WALLCLOCK": "time (Luo et al. 2014)",
 "TIMING_ASSERT": "time, async wait (Luo et al. 2014)",
 "RANDOM": "randomness (Gruber et al. 2021 add it as a category)",
 "CONCURRENCY": "concurrency, async wait (Luo et al. 2014)",
 "FIXED_PATH": "IO, test order dependency through shared files (Luo et al. 2014)",
 "SUBPROCESS": "IO, resource leak (Luo et al. 2014)",
 "UNORDERED": "unordered collections (Luo et al. 2014; NonDex, Gyori et al. 2016)",
 "NETWORK": "network (Luo et al. 2014; largest NOD category in Gruber et al. 2021)",
}

def checkout(row):
    name, url, sha, fn = row["Project_Name"], row["Project_URL"], row["Project_Hash"], row["Test_filename"]
    d = f"{REPOS}/{name}"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if not os.path.exists(f"{d}/.git"):
        os.makedirs(d, exist_ok=True)
        subprocess.run(["git", "init", "-q", d], env=env)
        subprocess.run(["git", "-C", d, "remote", "add", "origin", url], env=env)
    r = subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha],
                       capture_output=True, text=True, env=env, timeout=180)
    if r.returncode: return name, None
    r = subprocess.run(["git", "-C", d, "checkout", "-q", "FETCH_HEAD", "--", fn], capture_output=True, text=True, env=env, timeout=180)
    return name, (f"{d}/{fn}" if r.returncode == 0 and os.path.exists(f"{d}/{fn}") else None)

def tags_of(path):
    res = {}
    for t in smells.scan(path):
        tags = set(t["smells"])
        body = next(b for n, b, *_ in smells.test_functions(path) if n == t["test"])
        if NETWORK.search(body): tags.add("NETWORK")
        res[t["test"]] = tags
    return res

def main():
    rows = list(csv.DictReader(open(f"{V}/NOD_manual_classification.csv")))
    vb = list(csv.DictReader(open(f"{V}/victims_brittles.csv")))
    known = collections.defaultdict(set)
    for r in rows + vb:
        known[(r["Project_Name"], r["Test_filename"])].add(r["Test_funcname"])
    os.makedirs(REPOS, exist_ok=True)
    with ThreadPoolExecutor(12) as ex:
        paths = list(ex.map(checkout, rows))
    flaky, controls, missing = [], [], []
    seen_files = set()
    for row, (_, path) in zip(rows, paths):
        if not path:
            missing.append(row["Project_Name"]); continue
        tg = tags_of(path)
        fn = row["Test_funcname"]
        if fn not in tg:
            missing.append(row["Project_Name"] + "::" + fn); continue
        flaky.append({"project": row["Project_Name"], "test": fn, "category": row["flakiness category"], "tags": sorted(tg[fn])})
        key = (row["Project_Name"], row["Test_filename"])
        if key in seen_files: continue
        seen_files.add(key)
        for t, tags in tg.items():
            if t not in known[key]:
                controls.append({"project": row["Project_Name"], "test": t, "tags": sorted(tags)})
    base8 = lambda tags: [t for t in tags if t != "NETWORK"]
    def summary(items, f):
        return {"n": len(items), "flagged": sum(1 for x in items if f(x["tags"]))}
    by_cat = collections.defaultdict(list)
    for x in flaky: by_cat[x["category"]].append(x)
    res = {
        "rule_sources": RULE_SOURCES,
        "n_ground_truth": len(rows), "n_scanned": len(flaky), "missing": missing,
        "flaky_8_rules": summary(flaky, base8), "flaky_with_network": summary(flaky, lambda t: t),
        "controls_8_rules": summary(controls, base8), "controls_with_network": summary(controls, lambda t: t),
        "per_category": {c: {"n": len(v), "flagged_8": sum(1 for x in v if base8(x["tags"])),
                             "flagged_9": sum(1 for x in v if x["tags"]),
                             "rules": collections.Counter(t for x in v for t in x["tags"]).most_common()}
                         for c, v in sorted(by_cat.items(), key=lambda kv: -len(kv[1]))},
        "flaky": flaky,
    }
    json.dump(res, open(f"{V}/validation.json", "w"), indent=1)
    f8, f9, c8, c9 = res["flaky_8_rules"], res["flaky_with_network"], res["controls_8_rules"], res["controls_with_network"]
    print(f"scanned {len(flaky)} of {len(rows)} labeled flaky tests, {len(controls)} control tests from the same files")
    print(f"8 rules:      recall {f8['flagged']}/{f8['n']}   control flag rate {c8['flagged']}/{c8['n']}")
    print(f"+ NETWORK:    recall {f9['flagged']}/{f9['n']}   control flag rate {c9['flagged']}/{c9['n']}")
    for c, v in res["per_category"].items():
        print(f"  {c:22s} n={v['n']:3d} flagged(8)={v['flagged_8']:3d} flagged(9)={v['flagged_9']:3d} {v['rules'][:4]}")

if __name__ == "__main__":
    main()
