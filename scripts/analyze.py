#!/usr/bin/env python3
"""Aggregate every rerun into the study tables.

Suites: human, AG-S (agentic Sonnet, run-and-fix), OS-S (one-shot Sonnet), OS-H (one-shot Haiku).
A test is flaky if, across all reruns of its suite (all perturbation modes pooled), it both
passed and failed at least once. Always-fail tests are reported separately.
For one-shot suites every generated test is rerun, and each test is tagged with what a
single-run generate-and-filter step would have done with it (accepted or rejected).

Writes results/summary.json and prints markdown tables.
"""
import glob, json, math, os, collections

R = "/Users/shaunkapoor/ece382v-flaky-llm-tests/results"
PROJECTS = ["cachetools", "schedule", "tinydb", "humanize", "toolz", "tabulate",
            "tenacity", "filelock", "boltons", "more_itertools", "arrow", "pybreaker"]
PILOT = set(PROJECTS[:6])
HUMAN_NAME = {"tabulate": "python-tabulate", "more_itertools": "more-itertools"}
MODES = ["both", "tz", "stress"]
COND_LABEL = {"human": "Human", "llm_tests": "AG-S", "gen_oneshot_sonnet": "OS-S", "gen_oneshot_haiku": "OS-H"}


def run_files(cond, proj):
    out = {}
    for m in MODES:
        if cond == "human":
            p = f"{R}/human/{HUMAN_NAME.get(proj, proj)}.{m}.json"
        elif cond == "llm_tests" and proj in PILOT:
            p = f"{R}/llm/{proj}.{m}.json"
        else:
            p = f"{R}/runs/{cond}/{proj}.{m}.json"
        if os.path.exists(p):
            out[m] = json.load(open(p))
    return out


def is_pseudo(tid):
    """Collection-error entries are not tests."""
    cls, _, name = tid.partition("::")
    return cls in ("", "None") or name.endswith(".py") or name == ""


def merge(cond, proj):
    runs = run_files(cond, proj)
    if not runs:
        return None
    tests = collections.defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0, "modes_flaky": [], "modes_fail": [], "messages": []})
    for m, d in runs.items():
        for tid, v in d["tests"].items():
            if is_pseudo(tid):
                continue
            c = v["counts"]
            t = tests[tid]
            p, f = c.get("pass", 0), c.get("fail", 0) + c.get("error", 0)
            t["pass"] += p; t["fail"] += f; t["skip"] += c.get("skip", 0)
            if p and f: t["modes_flaky"].append(m)
            if f: t["modes_fail"].append(m)
            for msg in v.get("messages", []):
                if msg not in t["messages"] and len(t["messages"]) < 3:
                    t["messages"].append(msg)
    filt = None
    fp = f"{R}/filter/{cond}/{proj}.json"
    if os.path.exists(fp):
        filt = json.load(open(fp))
    acc = set(filt["accepted_ids"]) if filt else None
    for tid, t in tests.items():
        t["kind"] = ("flaky" if t["pass"] and t["fail"] else "always-fail" if t["fail"] else
                     "always-skip" if not t["pass"] else "stable")
        t["filter"] = None if acc is None else ("accepted" if tid in acc else "rejected")
    return {"modes_run": sorted(runs), "runs_per_mode": {m: d["runs"] for m, d in runs.items()},
            "tests": dict(tests), "filter": filt}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h) * 100, min(1, c + h) * 100)


def main():
    S = {}
    for cond in COND_LABEL:
        for proj in PROJECTS:
            m = merge(cond, proj)
            if m:
                S[(cond, proj)] = m
    # Table 1: per condition x project
    print("## Table A. Tests and flakiness per suite (all modes pooled)\n")
    print("| Subject | " + " | ".join(f"{COND_LABEL[c]} tests | {COND_LABEL[c]} flaky" for c in COND_LABEL) + " |")
    print("|---|" + "---|" * (2 * len(COND_LABEL)))
    for proj in PROJECTS:
        row = [proj]
        for c in COND_LABEL:
            m = S.get((c, proj))
            if not m:
                row += ["-", "-"]; continue
            ts = m["tests"].values()
            if m["filter"]:
                ts = [t for t in ts if t["filter"] == "accepted"]
            row += [str(len(ts)), str(sum(t["kind"] == "flaky" for t in ts))]
        print("| " + " | ".join(row) + " |")
    # Table 2: pooled by condition
    print("\n## Table B. Pooled flaky rate by condition (one-shot: accepted tests only)\n")
    print("| Condition | Suites | Tests | Flaky | Flaky % | 95% CI | Always fail |")
    print("|---|---|---|---|---|---|---|")
    pooled = {}
    for c in COND_LABEL:
        ms = [S[k] for k in S if k[0] == c]
        ts = [t for m in ms for t in m["tests"].values() if (not m["filter"] or t["filter"] == "accepted")]
        k = sum(t["kind"] == "flaky" for t in ts); n = len(ts); af = sum(t["kind"] == "always-fail" for t in ts)
        lo, hi = wilson(k, n)
        pooled[c] = {"suites": len(ms), "tests": n, "flaky": k, "always_fail": af, "ci": [lo, hi]}
        print(f"| {COND_LABEL[c]} | {len(ms)} | {n} | {k} | {100*k/max(n,1):.2f} | {lo:.2f} to {hi:.2f} | {af} |")
    # Table 3: what the single-run filter did, one-shot conditions
    print("\n## Table C. Single-run generate-and-filter on one-shot suites\n")
    print("| Condition | Collected tests | Files that do not import (test functions lost) | Accepted | Rejected | Rejected but flaky | Accepted but flaky | Accepted then always fail |")
    print("|---|---|---|---|---|---|---|---|")
    filt_tab = {}
    for c in ("gen_oneshot_sonnet", "gen_oneshot_haiku"):
        ms = [S[k] for k in S if k[0] == c and S[k]["filter"]]
        cef = sum(len(m["filter"]["collection_error_files"]) for m in ms)
        lost = 0  # test functions inside files that never import
        for (cc, proj), m in S.items():
            if cc != c or not m["filter"]:
                continue
            for f in m["filter"]["collection_error_files"]:
                p = f"{R}/../{c}/{proj}/{os.path.basename(f)}"
                if os.path.exists(p):
                    lost += sum(1 for line in open(p) if line.lstrip().startswith(("def test", "async def test")))
        ts = [t for m in ms for t in m["tests"].values()]
        acc = [t for t in ts if t["filter"] == "accepted"]; rej = [t for t in ts if t["filter"] == "rejected"]
        gen = len(acc) + len(rej)
        row = dict(generated=gen, collection_error_files=cef, lost_functions=lost, accepted=len(acc), rejected=len(rej),
                   rejected_flaky=sum(t["kind"] == "flaky" for t in rej),
                   accepted_flaky=sum(t["kind"] == "flaky" for t in acc),
                   accepted_always_fail=sum(t["kind"] == "always-fail" for t in acc))
        filt_tab[c] = row
        print(f"| {COND_LABEL[c]} | {gen} | {cef} ({lost}) | {len(acc)} | {len(rej)} | {row['rejected_flaky']} | {row['accepted_flaky']} | {row['accepted_always_fail']} |")
    # Table 4: which perturbation exposed each flaky test
    print("\n## Table D. Flaky tests and the modes in which they flipped\n")
    flaky_list = []
    for (c, proj), m in sorted(S.items()):
        for tid, t in m["tests"].items():
            if t["kind"] == "flaky":
                flaky_list.append({"cond": c, "proj": proj, "test": tid, "pass": t["pass"], "fail": t["fail"],
                                   "modes": t["modes_flaky"], "modes_fail": t["modes_fail"], "filter": t["filter"],
                                   "messages": t["messages"]})
                print(f"- {COND_LABEL[c]} / {proj} / `{tid}` pass={t['pass']} fail={t['fail']} "
                      f"flipped in {t['modes_flaky'] or t['modes_fail']} filter={t['filter']} :: {(t['messages'] or [''])[0][:140]}")
    if not flaky_list:
        print("(none)")
    json.dump({"pooled": pooled, "filter": filt_tab, "flaky": flaky_list,
               "suites": {f"{c}/{p}": {"tests": len(m["tests"]), "runs_per_mode": m["runs_per_mode"]} for (c, p), m in S.items()}},
              open(f"{R}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
