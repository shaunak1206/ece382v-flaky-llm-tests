#!/usr/bin/env python3
"""Model comparison tests for the matched study (E2), at the suite and project level.

Flaky tests cluster within suites and projects, so per test p values overstate the evidence. This script
compares models on two clustered units:
  per suite:   does a suite contain at least one kept flaky test? Fisher's exact test, two sided.
  per project: for each pair of models, count projects where only one of the two has a suite with a kept
               flaky test, and run an exact sign test (McNemar) on those discordant projects.
It reports the original 17 projects (the batch the Progress Report 2 draft was written on), the 24 projects
added on November 3 (a replication batch), and all of them. Reads results/matched/*.json plus the seeding
conftest runs in results/matched_mitig/, uses the same kept and polluter rules as analyze_matched.py, and
writes results/matched_stats.json.

  .venv/bin/python scripts/matched_stats.py
"""
import collections, glob, itertools, json, os
from scipy.stats import binomtest, fisher_exact

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, sys_path)
from analyze_matched import NEEDS_REJECTED_POLLUTER

ROOT = os.path.dirname(sys_path)
ORIGINAL = set(open(f"{ROOT}/results/gruber/matched_projects.txt").read().split()[:17])


def suite_rows():
    rows = []  # (project, condition, sid, kept, kept_flaky)
    for f in sorted(glob.glob(f"{ROOT}/results/matched/*.json")) + sorted(glob.glob(f"{ROOT}/results/matched_mitig/*.json")) + sorted(glob.glob(f"{ROOT}/results/matched_iso/*.json")):
        j = json.load(open(f))
        for sid, s in j["suites"].items():
            if s.get("prompt") not in ("plain", "plainseed", "plainiso"):
                continue
            if all(t.startswith("::") for t in s["tests"]):
                continue
            kept = [t for t in s["tests"] if t not in set(s["first_run_failed"])]
            fl = [t for t in kept if s["tests"][t]["kind"] == "flaky"
                  and (j["project"], sid, t.split("::")[-1]) not in NEEDS_REJECTED_POLLUTER]
            cond = s["model"] + {"plain": "", "plainseed": "+seed", "plainiso": "+iso"}[s["prompt"]]
            rows.append((j["project"], cond, sid, len(kept), len(fl)))
    return rows


def compare(rows, projects, models):
    out = {"projects": len(projects), "models": {}, "pairs": {}}
    by = collections.defaultdict(list)
    for p, c, sid, kept, fl in rows:
        if p in projects: by[c].append((p, kept, fl))
    for m in models:
        r = by[m]
        out["models"][m] = {"suites": len(r), "suites_with_flaky": sum(1 for _, _, f in r if f),
                            "projects_with_flaky": len({p for p, _, f in r if f}),
                            "kept": sum(k for _, k, _ in r), "flaky_kept": sum(f for _, _, f in r)}
    for a, b in itertools.combinations(models, 2):
        A, B = out["models"][a], out["models"][b]
        _, p_suite = fisher_exact([[A["suites_with_flaky"], A["suites"] - A["suites_with_flaky"]],
                                   [B["suites_with_flaky"], B["suites"] - B["suites_with_flaky"]]])
        pa = {p for p, _, f in by[a] if f}; pb = {p for p, _, f in by[b] if f}
        both = {p for p, _, _ in by[a]} & {p for p, _, _ in by[b]}
        only_a, only_b = len((pa - pb) & both), len((pb - pa) & both)
        p_proj = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        out["pairs"][f"{a} vs {b}"] = {"p_suite_fisher": round(p_suite, 4), "projects_only_first": only_a,
                                        "projects_only_second": only_b, "p_project_sign": round(p_proj, 4)}
    return out


def main():
    rows = suite_rows()
    allp = {p for p, *_ in rows if not p.startswith("::")}
    models = ["sonnet", "haiku", "gpt54mini", "gpt41mini"]
    res = {}
    for name, ps in (("original_17", ORIGINAL & allp), ("new_24", allp - ORIGINAL), ("all", allp)):
        res[name] = compare(rows, ps, models)
        res[name + "_seeded"] = compare(rows, ps, [m + "+seed" for m in models])
        res[name + "_isolated"] = compare(rows, ps, [m + "+iso" for m in models])
    json.dump(res, open(f"{ROOT}/results/matched_stats.json", "w"), indent=1)
    for name in ("original_17", "new_24", "all"):
        r = res[name]
        print(f"\n## {name} ({r['projects']} projects)")
        for m, v in r["models"].items():
            print(f"  {m:10s} suites {v['suites']:3d}  with flaky {v['suites_with_flaky']:2d}  projects {v['projects_with_flaky']:2d}  "
                  f"flaky/kept {v['flaky_kept']}/{v['kept']}")
        for k, v in r["pairs"].items():
            if "gpt41mini" in k: print(f"  {k:24s} suite p={v['p_suite_fisher']}  projects {v['projects_only_first']}:{v['projects_only_second']} sign p={v['p_project_sign']}")
        s = res[name + "_seeded"]
        print("  seeded:", {m: (v["suites_with_flaky"], v["suites"], v["flaky_kept"]) for m, v in s["models"].items()})
        s = res[name + "_isolated"]
        print("  isolated:", {m: (v["suites_with_flaky"], v["suites"], v["flaky_kept"]) for m, v in s["models"].items()})


if __name__ == "__main__":
    main()
