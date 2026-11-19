#!/usr/bin/env python3
"""Failure rate of the flaky tests each mitigation leaves (Progress Report 2, step 2).

For the held out sample, per model and condition (plain, M1, M2): the kept flaky tests and their failure rate over
the 200 runs (fails and errors over runs where the test ran), as the median and the share of tests that fail in at
least 5% of runs. A filter or a developer sees a rarely failing test less often, so a mitigation that leaves only
rare flips helps even when it does not remove them. Writes results/leftover_rates.json.

  python scripts/leftover_rates.py
"""
import glob, json, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rates = {}
    for f in glob.glob(f"{ROOT}/results/matched_heldout/*.json"):
        for sid, s in json.load(open(f))["suites"].items():
            m, cond, _ = sid.split("_")
            if all(t.startswith("::") for t in s["tests"]): continue
            first = set(s["first_run_failed"])
            for t, v in s["tests"].items():
                if t in first or v["kind"] != "flaky": continue
                n = v["pass"] + v["fail"] + v["error"]
                rates.setdefault(f"{m}_{cond}", []).append((v["fail"] + v["error"]) / n)
    out = {k: {"flaky_kept": len(r), "median_fail_rate": round(st.median(r), 3), "mean_fail_rate": round(st.mean(r), 3),
               "share_at_least_5pct": round(sum(x >= 0.05 for x in r) / len(r), 2)} for k, r in sorted(rates.items())}
    json.dump(out, open(f"{ROOT}/results/leftover_rates.json", "w"), indent=1)
    for k, v in out.items(): print(k, v)


if __name__ == "__main__":
    main()
