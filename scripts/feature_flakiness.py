#!/usr/bin/env python3
"""Which nondeterminism features do human and LLM tests use, and do the tests that use them flip? (E1, 12 libraries)

Joins the per test function features of scripts/characterize.py (results/characterize.json, "tests") with the
flaky tests of the large rerun campaign (results/scale_summary.json, "flaky"). A test function counts as flaky when
any of its cases flipped. For each feature it reports, for the human suites and for all LLM suites together, how
many test functions use it, the share of functions that use it, and how many of those functions flipped.

Writes results/feature_flakiness.json and prints a markdown table.
"""
import json, os
from collections import defaultdict

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = [  # (key in characterize.json, label, group)
    ("random_calls", "Calls a random number generator", "randomness"),
    ("random_unseeded", "Uses randomness without a seed", "randomness"),
    ("real_clock_read", "Reads the real clock", "time"),
    ("real_sleep", "Sleeps for real", "time"),
    ("timing_assert", "Asserts on elapsed time", "time"),
    ("concurrency", "Uses threads, processes, or asyncio", "concurrency"),
    ("threads_or_procs_without_join", "Starts a thread or process without join", "concurrency"),
    ("shared_state_any", "Shares state between tests", "order"),
    ("wide_scope_fixture", "Class or module scoped fixture", "order"),
    ("setup_without_teardown", "Setup without teardown", "order"),
    ("iso_freezegun", "Freezes the clock (freezegun)", "isolation"),
    ("iso_monkeypatch", "Uses monkeypatch", "isolation"),
]


def main():
    c = json.load(open(f"{R}/results/characterize.json"))
    recs = c["tests"]
    flaky = set()
    for t in json.load(open(f"{R}/results/scale_summary.json"))["flaky"]:
        cls_path, name = t["test"].split("::", 1)
        flaky.add((t["cond"], t["proj"], name.split("[")[0], cls_path.split(".")[-1]))
    def is_flaky(r):
        return any(k[0] == r["cond"] and k[1] == r["proj"] and k[2] == r["test"] and k[3] in (r.get("class") or "", os.path.basename(r["file"])[:-3])
                   for k in flaky)
    groups = {"human": [r for r in recs if r["cond"] == "human"], "llm": [r for r in recs if r["cond"] != "human"]}
    out = {"features": {}, "totals": {}}
    for g, rs in groups.items():
        fl = [r for r in rs if is_flaky(r)]
        out["totals"][g] = {"functions": len(rs), "flaky_functions": len(fl)}
    for key, label, grp in FEATURES:
        row = {"label": label, "group": grp}
        for g, rs in groups.items():
            use = [r for r in rs if r.get(key)]
            fl = [r for r in use if is_flaky(r)]
            row[g] = {"use": len(use), "share": round(100 * len(use) / len(rs), 1), "flaky": len(fl),
                      "flaky_tests": sorted(f"{r['cond']}/{r['proj']}::{r['test']}" for r in fl)}
        out["features"][key] = row
    json.dump(out, open(f"{R}/results/feature_flakiness.json", "w"), indent=1)
    print(f"| Feature | Human use (% of {out['totals']['human']['functions']}) | Human flaky | LLM use (% of {out['totals']['llm']['functions']}) | LLM flaky |")
    print("|---|---|---|---|---|")
    for key, row in out["features"].items():
        h, l = row["human"], row["llm"]
        print(f"| {row['label']} | {h['use']} ({h['share']}) | {h['flaky']} | {l['use']} ({l['share']}) | {l['flaky']} |")
    print("totals", out["totals"])


if __name__ == "__main__":
    main()
