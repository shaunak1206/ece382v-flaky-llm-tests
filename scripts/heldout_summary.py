#!/usr/bin/env python3
"""Counts for the held out sample and the Claude guard v3 suites (Progress Report 3).

Reads results/matched_heldout/*.json, results/matched_od/*.json (order dependent arm, reported as od, odseed, odiso) (suites <model>_heldout_<k>, and their copies with fixture M1,
<model>_heldoutseed_<k>, and M2, <model>_heldoutiso_<k>) and results/matched_guard3cl/*.json (Claude guard v3), all
written by scripts/matched_run.py with 200 runs. The same counting rules as analyze_matched.py: a test is kept if it
passed the first run, flaky if it passed and failed. For each model and condition it reports suites, suites that did
not import, kept tests, kept flaky tests, suites and projects with a kept flaky test, and how many kept flaky tests
fail only in the shuffled runs (order dependent for a file order filter).
Fixture cost: tests that are kept and never fail without the fixture but fail in every run with it.
Writes results/heldout_summary.json.

  python scripts/heldout_summary.py
"""
import collections, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def suite_stats(s):
    if all(t.startswith("::") for t in s["tests"]):
        return None
    first = set(s["first_run_failed"])
    kept = {t: v for t, v in s["tests"].items() if t not in first}
    fl = {t: v for t, v in kept.items() if v["kind"] == "flaky"}
    return kept, fl


def main():
    out = {"groups": {}, "fixture_cost": {}, "flaky": []}
    G = collections.defaultdict(collections.Counter)
    gproj = collections.defaultdict(set)
    per = {}  # (project, model, k, cond) -> tests
    for d in ("matched_heldout", "matched_guard3cl", "matched_od"):
        for f in sorted(glob.glob(f"{ROOT}/results/{d}/*.json")):
            j = json.load(open(f))
            for sid, s in j["suites"].items():
                m, cond, k = sid.split("_")
                if d == "matched_od": cond = cond.replace("plain", "od")  # order dependent arm: od, odseed, odiso
                key = f"{m}_{cond}"
                g = G[key]
                r = suite_stats(s)
                g["suites"] += 1
                if r is None:
                    g["not_imported"] += 1; continue
                kept, fl = r
                per[(j["project"], m, k, cond)] = s["tests"]
                g["generated"] += len(s["tests"]); g["kept"] += len(kept); g["flaky_kept"] += len(fl)
                g["removed_flaky"] += sum(1 for t, v in s["tests"].items() if t in set(s["first_run_failed"]) and v["kind"] == "flaky")
                if fl:
                    g["suites_with_flaky"] += 1; gproj[key].add(j["project"])
                for t, v in fl.items():
                    fo = v.get("fails_file_order"); sh = v.get("fails_shuffled")
                    shuffle_only = fo == 0 and (sh or 0) > 0
                    g["shuffle_only"] += shuffle_only
                    out["flaky"].append({"project": j["project"], "suite": sid, "arm": d, "test": t, "pass": v["pass"],
                                         "fail": v["fail"] + v["error"], "fails_file_order": fo, "fails_shuffled": sh,
                                         "first_message": v.get("first_message", "")[:200]})
    for key, g in G.items():
        g["projects_with_flaky"] = len(gproj[key])
        out["groups"][key] = dict(g)
    # fixture cost and effect, paired on the same suite
    for base, fx in (("heldout", "heldoutseed"), ("heldout", "heldoutiso"), ("od", "odseed"), ("od", "odiso")):
        c = collections.Counter()
        for (p, m, k, cond), tests in per.items():
            if cond != base or (p, m, k, fx) not in per: continue
            other = per[(p, m, k, fx)]
            for t, v in tests.items():
                t2 = t.replace(f"_{'plain' if base == 'od' else base}_{k}.", f"_{fx.replace('od', 'plain') if base == 'od' else fx}_{k}.")
                w = other.get(t2)
                if w is None: continue
                if v["kind"] == "stable-pass" or (v["fail"] + v["error"] == 0 and v["pass"] > 0):
                    if w["pass"] == 0 and (w["fail"] + w["error"]) > 0:
                        c[f"{m}_broken"] += 1; out.setdefault("broken", []).append({"project": p, "suite": f"{m}_{fx}_{k}", "test": t2})
                if v["kind"] == "flaky":
                    c[f"{m}_flaky_before"] += 1
                    c[f"{m}_still_flaky" if w["kind"] == "flaky" else f"{m}_not_flaky_after"] += 1
        out["fixture_cost"][fx] = dict(c)
    json.dump(out, open(f"{ROOT}/results/heldout_summary.json", "w"), indent=1)
    for k in sorted(out["groups"]): print(k, out["groups"][k])
    print(json.dumps(out["fixture_cost"], indent=1))


if __name__ == "__main__":
    main()
