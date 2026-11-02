#!/usr/bin/env python3
"""Tables for the matched study: LLM tests written for code that is known to cause a flaky human test.

Reads results/matched/<project>.json (from scripts/matched_run.py), writes results/matched_summary.json and
prints markdown tables. A generated test is counted only if a one run filter would keep it (it passed in the
first run), the same rule as the 12 library study; always failing tests are reported separately.
"""
import collections, glob, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = {"sonnet": "Claude Sonnet", "haiku": "Claude Haiku 4.5", "gpt54mini": "GPT 5.4 mini", "gpt41mini": "GPT 4.1 mini"}


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h) * 100, min(1, c + h) * 100)


def main():
    files = sorted(glob.glob(f"{ROOT}/results/matched/*.json"))
    per_model = collections.defaultdict(lambda: collections.Counter())
    per_project = []
    flaky_list = []
    for f in files:
        j = json.load(open(f))
        h = j.get("human", {})
        row = {"project": j["project"], "category": j["category"], "human_runs": h.get("runs", 0),
               "human_failures": h.get("failed", 0) + h.get("errors", 0), "human_reproduced": h.get("reproduced", False)}
        for sid, s in j["suites"].items():
            m = s["model"]
            # a module level id (classname::file) means the file did not import, so no test ran
            if all(t.startswith("::") for t in s["tests"]):
                per_model[m]["suites_not_collected"] += 1
                continue
            kept = {t: v for t, v in s["tests"].items() if t not in set(s["first_run_failed"])}
            fl = [t for t, v in kept.items() if v["kind"] == "flaky"]
            pm = per_model[m]
            pm["suites"] += 1; pm["generated"] += len(s["tests"]); pm["kept"] += len(kept); pm["flaky"] += len(fl)
            pm["rejected"] += len(s["first_run_failed"])
            pm["rejected_but_flaky"] += sum(1 for t in s["first_run_failed"] if s["tests"].get(t, {}).get("kind") == "flaky")
            pm["suites_with_flaky"] += 1 if fl else 0
            row[f"{sid}_flaky"] = len(fl)
            for t in fl:
                v = kept[t]
                flaky_list.append({"project": j["project"], "suite": sid, "test": t, "pass": v["pass"],
                                   "fail": v["fail"] + v["error"], "category": j["category"]})
        per_project.append(row)
    print("## Matched study, by model (tests a one run filter keeps)\n")
    print("| Model | Suites | Tests generated | Kept after one run | Flaky kept tests | Flaky % (95% CI) | Suites with a flaky test | Rejected by the filter | Rejected but flaky | Suites that did not import |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    out_models = {}
    for m, c in per_model.items():
        lo, hi = wilson(c["flaky"], c["kept"])
        out_models[m] = dict(c, ci=[lo, hi])
        print(f"| {MODEL.get(m, m)} | {c['suites']} | {c['generated']} | {c['kept']} | {c['flaky']} | "
              f"{100 * c['flaky'] / max(c['kept'], 1):.2f} ({lo:.2f} to {hi:.2f}) | {c['suites_with_flaky']} | {c['rejected']} | {c['rejected_but_flaky']} | {c['suites_not_collected']} |")
    print("\n## Human control per project (the known flaky test, same runner, same number of runs)\n")
    print("| Project | Gruber category | Runs | Failures | Reproduced |")
    print("|---|---|---|---|---|")
    for r in per_project:
        print(f"| {r['project']} | {r['category']} | {r['human_runs']} | {r['human_failures']} | {'yes' if r['human_reproduced'] else 'no'} |")
    print("\n## Flaky generated tests\n")
    for x in flaky_list:
        print(f"- {x['project']} / {x['suite']} / `{x['test']}` pass={x['pass']} fail={x['fail']}")
    if not flaky_list: print("(none)")
    json.dump({"models": out_models, "projects": per_project, "flaky": flaky_list},
              open(f"{ROOT}/results/matched_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
