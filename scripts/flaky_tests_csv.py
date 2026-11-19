#!/usr/bin/env python3
"""Write results/flaky_tests.csv: every flaky generated test of the known flaky code study, one row each.

A test is flaky if it both passed and failed in the 200 runs of its suite (results/matched*, as matched_run.py
records it). Columns: arm (held out, guard, order dependent arm, Progress Report 2 suites), project, model, prompt,
suite, test file, test id, kept (passed the filter run), runs, failures, failure rate, failures in file order and in
shuffled runs, and, where the polluter search ran, its verdict and polluters, and whether a 10 run rerun confirmed the
polluter. M1/M2 copies are left out (they rerun the same tests).

  python scripts/flaky_tests_csv.py
"""
import csv, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = [("matched_heldout", "held out"), ("matched_od", "order dependent arm"), ("matched_guard3cl", "guard v3"),
        ("matched_guard3gpt", "guard v3"), ("matched", "Progress Report 2"), ("matched_guard", "Progress Report 2"),
        ("matched_guard3", "Progress Report 2"), ("matched_mitig", "Progress Report 2")]


def main():
    diag, conf = {}, {}
    for f in glob.glob(f"{ROOT}/results/od_diagnose*/*.json"):
        j = json.load(open(f))
        for t in j.get("tests", []):
            diag[(j["project"], t["suite"], t["test"])] = (t["verdict"], ";".join(t["polluters"]))
    for f in glob.glob(f"{ROOT}/results/cleaners/*.json"):
        j = json.load(open(f))
        for v in j["victims"]:
            if "confirmed" in v: conf[(j["project"], v["suite"], v["test"])] = v["confirmed"]
    rows, seen = [], set()
    for d, arm in ARMS:
        for f in sorted(glob.glob(f"{ROOT}/results/{d}/*.json")):
            j = json.load(open(f)); p = j["project"]
            for sid, s in sorted(j["suites"].items()):
                if s.get("prompt", "").endswith(("seed", "iso")) or (p, sid) in seen: continue
                seen.add((p, sid))
                first = set(s["first_run_failed"])
                for t, v in sorted(s["tests"].items()):
                    if v["kind"] != "flaky": continue
                    name = t.split("::")[-1]; n = v["pass"] + v["fail"] + v["error"]
                    verdict, pols = diag.get((p, sid, name), ("", ""))
                    c = conf.get((p, sid, name))
                    rows.append({"arm": arm, "project": p, "model": s.get("model", ""), "prompt": s.get("prompt", ""),
                                 "suite": sid, "test_file": f"matched/{p}/{sid}/test_gen.py", "test_id": t,
                                 "kept": t not in first, "runs": n, "failures": v["fail"] + v["error"],
                                 "failure_rate": round((v["fail"] + v["error"]) / n, 3) if n else "",
                                 "fails_file_order": v.get("fails_file_order", ""), "fails_shuffled": v.get("fails_shuffled", ""),
                                 "polluter_search": verdict, "polluters": pols,
                                 "polluter_confirmed": "" if c is None else c})
    out = f"{ROOT}/results/flaky_tests.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    kept = sum(r["kept"] for r in rows)
    print(f"{len(rows)} flaky tests ({kept} kept) -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
