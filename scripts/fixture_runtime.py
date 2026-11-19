#!/usr/bin/env python3
"""Run time cost of the fixtures, from the CI logs of the held out runs (Progress Report 3).

results/fixture_runtime/by_job.txt holds, for every job of the six held out matched-study runs (plain, M1, M2; run
ids in logs/runs_gpt_pr3.txt), the timestamp of each "[project/suite] tests=..." line matched_run.py prints when a
suite finishes its 200 runs. A suite's time is the gap to the previous line of the same job (the first suite of a job,
which also includes setup, is left out). We compare each plain suite with its M1 and M2 copy.

  python scripts/fixture_runtime.py
"""
import collections, datetime, json, os, re, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    dur, prev = {}, None
    for line in open(f"{ROOT}/results/fixture_runtime/by_job.txt"):
        if line.startswith("=="): prev = None; continue
        m = re.match(r"(\S+)Z \[([^/]+)/([a-z0-9]+)_(heldout(?:seed|iso)?)_(\d)\]", line)
        if not m: continue
        t = datetime.datetime.fromisoformat(m.group(1)[:26])
        if prev is not None: dur[(m.group(2), m.group(3), m.group(5), m.group(4))] = (t - prev).total_seconds()
        prev = t
    out = {}
    for fx, name in (("heldoutseed", "M1"), ("heldoutiso", "M2")):
        pairs = [(d, dur[(p, m, k, fx)]) for (p, m, k, c), d in dur.items() if c == "heldout" and (p, m, k, fx) in dur and d > 0]
        out[name] = {"suite_pairs": len(pairs), "plain_hours": round(sum(a for a, _ in pairs) / 3600, 1),
                     "fixture_hours": round(sum(b for _, b in pairs) / 3600, 1),
                     "total_ratio": round(sum(b for _, b in pairs) / sum(a for a, _ in pairs), 3),
                     "median_ratio": round(st.median(b / a for a, b in pairs), 3)}
    json.dump(out, open(f"{ROOT}/results/fixture_runtime.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
