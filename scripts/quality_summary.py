#!/usr/bin/env python3
"""Summarize results/quality*/ (scripts/e2_quality.py) per model and prompt (Progress Report 3).

For each model and prompt: suites, generated tests, kept tests (pass the clean file order run), suites with no kept
test, mean line coverage of the target files, and the mutation score pooled over suites (killed / mutants run). A
suite with no kept test kills nothing, so it counts as 0 killed out of the project's mutants. Mutant counts come from
the project's fixed mutant set, so pooled scores of two prompts on the same projects face the same mutants.
Prompt comparisons are paired: only projects that have both prompts for that model, and again only the suite pairs
(same project and sample) in which both suites keep at least one test.
Writes results/quality_summary.json.

  python scripts/quality_summary.py [results/quality results/quality_heldout ...]
"""
import glob, json, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.json")):
            j = json.load(open(f))
            nm = len(j["mutants"])
            for sid, s in j["suites"].items():
                model, prompt = sid.split("_")[0], sid.split("_")[1]
                rows.append({"project": j["project"], "suite": sid, "model": model, "prompt": prompt, "tests": s["tests"],
                             "kept": s["passing"], "coverage": s["coverage"] or 0.0,
                             "killed": s["killed"], "mutants": s["mutants"] if s["passing"] else nm})
    return rows


def agg(rs):
    m = sum(r["mutants"] for r in rs)
    return {"suites": len(rs), "projects": len({r["project"] for r in rs}), "tests": sum(r["tests"] for r in rs),
            "kept": sum(r["kept"] for r in rs), "no_kept": sum(r["kept"] == 0 for r in rs),
            "coverage_mean": round(st.mean(r["coverage"] for r in rs), 1) if rs else None,
            "mutation_pooled": round(100 * sum(r["killed"] for r in rs) / m, 1) if m else None}


def main():
    dirs = sys.argv[1:] or [f"{ROOT}/results/quality"]
    rows = load(dirs)
    groups = sorted({(r["model"], r["prompt"]) for r in rows})
    out = {"groups": {f"{m}_{p}": agg([r for r in rows if (r["model"], r["prompt"]) == (m, p)]) for m, p in groups},
           "paired": {}}
    for m, p in groups:
        base = "plain" if p != "heldout" else None
        for q in ("plain", "heldout"):
            if q == p or (m, q) not in groups:
                continue
            projs = {r["project"] for r in rows if (r["model"], r["prompt"]) == (m, p)} & \
                    {r["project"] for r in rows if (r["model"], r["prompt"]) == (m, q)}
            # suite pairs (same project and sample) where both suites keep at least one test
            pr = {}
            for r in rows:
                if r["model"] == m and r["prompt"] in (p, q):
                    pr.setdefault((r["project"], r["suite"].split("_")[-1]), {})[r["prompt"]] = r
            both = [v for v in pr.values() if p in v and q in v and v[p]["kept"] and v[q]["kept"]]
            imported = {x: agg([v[x] for v in both]) for x in (p, q)}
            out["paired"][f"{m}: {p} vs {q}, both keep tests"] = imported
            out["paired"][f"{m}: {p} vs {q}"] = {
                p: agg([r for r in rows if (r["model"], r["prompt"]) == (m, p) and r["project"] in projs]),
                q: agg([r for r in rows if (r["model"], r["prompt"]) == (m, q) and r["project"] in projs])}
    name = "quality_summary.json" if len(dirs) == 1 and dirs[0].endswith("results/quality") else "quality_summary_all.json"
    json.dump(out, open(f"{ROOT}/results/{name}", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
