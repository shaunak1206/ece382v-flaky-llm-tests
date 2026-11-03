#!/usr/bin/env python3
"""Expected share of the kept flaky LLM tests that a filter with k extra runs would catch.

A test that fails in a fraction f of runs is caught by k extra runs with probability 1 - (1 - f)^k. The estimate
is made twice: extra runs in file order (what a one run filter repeated would do) and extra runs in shuffled
order. f comes from the measured failures of each kept flaky test:
  E1: results/scale_summary.json (every E1 mode shuffles). The PL-G4 humanize tests are order dependent and pass
      in file order (results/why_analysis.md), so their file order f is 0; for the others f is the same.
  E2: results/matched*/ plain, guard, guard2 and guard3 suites (not the M1 and M2 copies, which rerun the same
      tests), with the file order and shuffled failure counts that matched_run.py records. The original 17
      projects ran before those counts existed; their kept flaky tests are not order dependent, so f is pooled.
Writes results/krun_estimate.json.
"""
import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matched import NEEDS_REJECTED_POLLUTER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS = (1, 3, 5, 10)


def tests():
    out = []  # (source, f_file, f_shuffled)
    for t in json.load(open(f"{ROOT}/results/scale_summary.json"))["flaky"]:
        if t["cond"] == "human": continue
        f = t["fail"] / (t["pass"] + t["fail"])
        od = t["cond"] == "gen_plain_gpt41mini" and t["proj"] == "humanize"
        out.append(("E1", 0.0 if od else f, f))
    for path in glob.glob(f"{ROOT}/results/matched*/*.json"):
        j = json.load(open(path))
        for sid, s in j["suites"].items():
            if s.get("prompt") in ("plainseed", "plainiso"): continue
            rejected = set(s["first_run_failed"])
            for t, v in s["tests"].items():
                if v["kind"] != "flaky" or t in rejected: continue
                if (j["project"], sid, t.split("::")[-1]) in NEEDS_REJECTED_POLLUTER: continue
                n = v["pass"] + v["fail"] + v["error"]; fails = v["fail"] + v["error"]
                if v.get("fails_file_order") is None:
                    out.append(("E2", fails / n, fails / n))
                else:
                    half = s["runs"] / 2
                    out.append(("E2", v["fails_file_order"] / half, v["fails_shuffled"] / half))
    return out


def main():
    ts = tests()
    res = {"tests": len(ts), "file_order": {}, "shuffled": {},
           "fail_under_5pct": sum(1 for _, a, b in ts if max(a, b) < 0.05),
           # over all runs (E2 runs are half file order, half shuffled; E1 b is already the all-runs rate)
           "fail_under_5pct_all_runs": sum(1 for src, a, b in ts if ((a + b) / 2 if src == "E2" else b) < 0.05),
           "never_fail_in_file_order": sum(1 for _, a, _ in ts if a == 0)}
    for k in KS:
        res["file_order"][k] = round(100 * sum(1 - (1 - a) ** k for _, a, _ in ts) / len(ts), 1)
        res["shuffled"][k] = round(100 * sum(1 - (1 - b) ** k for _, _, b in ts) / len(ts), 1)
    json.dump(res, open(f"{ROOT}/results/krun_estimate.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
