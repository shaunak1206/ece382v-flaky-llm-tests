#!/usr/bin/env python3
"""Tables for the large rerun campaign (results/runs_scale/, all on 2 vCPU Ubuntu GitHub runners).

A test is flaky if it passed and failed across the campaign runs of its suite (one machine type, all modes
pooled). Generated tests count only if the one run filter kept them (results/filter/), the same rule as
analyze.py. Writes results/scale_summary.json and prints markdown tables.
"""
import collections, glob, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{ROOT}/results"
LABEL = {"human": "Human", "llm_tests": "AG-S", "gen_oneshot_sonnet": "OS-S", "gen_oneshot_haiku": "OS-H",
         "gen_tp_sonnet": "TP-S", "gen_plain_sonnet": "PL-S", "gen_plain_gpt54mini": "PL-G5", "gen_plain_gpt41mini": "PL-G4"}
HUMAN_NAME = {"tabulate": "python-tabulate", "more_itertools": "more-itertools"}


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h) * 100, min(1, c + h) * 100)


def pseudo(tid):
    cls, _, name = tid.partition("::")
    return cls in ("", "None") or name.endswith(".py") or name == ""


def main():
    pooled, flaky, runs_tot = {}, [], collections.Counter()
    for cond in LABEL:
        tests_all = 0; n_flaky = 0; af = 0; suites = 0
        for proj_file in sorted(glob.glob(f"{R}/runs_scale/{cond}/*.both.json")):
            proj = os.path.basename(proj_file).split(".")[0]
            agg = collections.defaultdict(collections.Counter); modes = {}
            for m in ("both", "tz", "stressng"):
                p = f"{R}/runs_scale/{cond}/{proj}.{m}.json"
                if not os.path.exists(p): continue
                d = json.load(open(p)); modes[m] = d["runs"]; runs_tot[cond] += d["runs"] * (d["num_tests"] or 0)
                for tid, v in d["tests"].items():
                    if pseudo(tid): continue
                    c = v["counts"]; agg[tid]["pass"] += c.get("pass", 0); agg[tid]["fail"] += c.get("fail", 0) + c.get("error", 0)
                    if c.get("fail", 0) + c.get("error", 0) and c.get("pass", 0): agg[tid]["modes_" + m] += 1
            keep = None
            fp = f"{R}/filter/{cond}/{proj}.json"
            if cond != "human" and os.path.exists(fp):
                keep = set(json.load(open(fp))["accepted_ids"])
            suites += 1
            for tid, c in agg.items():
                if keep is not None and tid not in keep: continue
                tests_all += 1
                if c["pass"] and c["fail"]:
                    n_flaky += 1
                    flaky.append({"cond": cond, "proj": proj, "test": tid, "pass": c["pass"], "fail": c["fail"],
                                  "rate": round(c["fail"] / (c["pass"] + c["fail"]), 4),
                                  "modes": [m for m in ("both", "tz", "stressng") if c["modes_" + m]], "runs": modes})
                elif c["fail"] and not c["pass"]:
                    af += 1
        lo, hi = wilson(n_flaky, tests_all)
        pooled[cond] = {"suites": suites, "tests": tests_all, "flaky": n_flaky, "always_fail": af, "ci": [lo, hi]}
    print("## Large rerun campaign (2 vCPU Ubuntu runners)\n")
    print("| Condition | Suites | Tests | Runs per test | Flaky | Flaky % (95% CI) | Always fail |")
    print("|---|---|---|---|---|---|---|")
    for cond, p in pooled.items():
        runs = "200" if cond == "human" else "600"
        print(f"| {LABEL[cond]} | {p['suites']} | {p['tests']} | {runs} | {p['flaky']} | {100 * p['flaky'] / max(p['tests'], 1):.2f} "
              f"({p['ci'][0]:.2f} to {p['ci'][1]:.2f}) | {p['always_fail']} |")
    print(f"\nTest executions: {sum(runs_tot.values()):,}\n")
    print("## Flaky tests in the campaign\n")
    for x in flaky:
        print(f"- {LABEL[x['cond']]} / {x['proj']} / `{x['test']}` pass={x['pass']} fail={x['fail']} ({100 * x['rate']:.1f}%) modes={x['modes']}")
    json.dump({"pooled": pooled, "flaky": flaky, "executions": dict(runs_tot)}, open(f"{R}/scale_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
