#!/usr/bin/env python3
"""Stress-ng reruns of the six known flaky code projects outside randomness (Progress Report 3, goal 4).

Compares, for the held out plain suites of each project and model, the 150 runs under stress-ng
(results/matched_stress) with the same suites' 200 plain runs (results/matched_heldout). Kept means passed run 0 of
that run set. Writes results/stress_summary.json.

  python scripts/stress_summary.py
"""
import collections, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stats(s):
    if all(t.startswith("::") for t in s["tests"]): return None
    first = set(s["first_run_failed"])
    kept = {t: v for t, v in s["tests"].items() if t not in first}
    return kept, {t for t, v in kept.items() if v["kind"] == "flaky"}


def main():
    out = {"by_project": {}, "by_model": collections.defaultdict(lambda: collections.Counter()), "new_flaky": []}
    for f in sorted(glob.glob(f"{ROOT}/results/matched_stress/*.json")):
        st = json.load(open(f)); p = st["project"]
        pl = json.load(open(f"{ROOT}/results/matched_heldout/{p}.json"))["suites"]
        row = collections.Counter()
        for sid, s in st["suites"].items():
            a, b = stats(s), stats(pl[sid]) if sid in pl else None
            if not a or not b: continue
            m = s["model"]; c = out["by_model"][m]
            c["suites"] += 1; row["suites"] += 1
            c["stress_suites_with_flaky"] += bool(a[1]); c["plain_suites_with_flaky"] += bool(b[1])
            row["stress_suites_with_flaky"] += bool(a[1]); row["plain_suites_with_flaky"] += bool(b[1])
            c["stress_flaky"] += len(a[1]); c["plain_flaky"] += len(b[1]); c["kept_stress"] += len(a[0])
            row["stress_flaky"] += len(a[1]); row["plain_flaky"] += len(b[1])
            for t in a[1] - b[1]:
                out["new_flaky"].append({"project": p, "suite": sid, "test": t.split("::")[-1],
                                         "plain_kind": pl[sid]["tests"].get(t, {}).get("kind"),
                                         "stress_fail": s["tests"][t]["fail"] + s["tests"][t]["error"],
                                         "message": (s["tests"][t].get("first_message") or "")[:160]})
            # same run count as the stress rerun: the first 150 plain runs (run order sequences)
            def wf150(S):
                first = set(S["first_run_failed"])
                return any(("P" in v["sequence"][:150]) and (set(v["sequence"][:150]) & {"F", "E"})
                           for t, v in S["tests"].items() if t not in first)
            row["plain_first150_suites_with_flaky"] += wf150(pl[sid])
        out["by_project"][p] = dict(row)
    out["by_model"] = {k: dict(v) for k, v in out["by_model"].items()}
    out["totals"] = {k: sum(r.get(k, 0) for r in out["by_project"].values()) for k in
                     ("suites", "stress_suites_with_flaky", "plain_suites_with_flaky", "plain_first150_suites_with_flaky")}
    json.dump(out, open(f"{ROOT}/results/stress_summary.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("totals", "by_model")}, indent=1))
    for x in out["new_flaky"]: print(x)


if __name__ == "__main__":
    main()
