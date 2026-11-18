#!/usr/bin/env python3
"""Every number in the Progress Report 3 tables that comes from the 200 run results, in one place.

Reads results/matched_heldout, matched_od, matched_guard3cl, matched_guard3gpt (written by matched_run.py, joined by
join_runs.py where a suite was split), and results/quality_summary_all.json for coverage and mutation score.
Counting rules are those of heldout_summary.py: a suite imported if any test was collected, a test kept if it passed
run 0, flaky if it passed and failed in the 200 runs. Prints table rows and per project breakdowns; writes
results/report3_numbers.json.

  python scripts/report3_numbers.py
"""
import collections, glob, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["sonnet", "haiku", "gpt54mini", "gpt41mini"]
NAME = {"sonnet": "Sonnet 5.5", "haiku": "Haiku 5.5", "gpt54mini": "GPT 5.4 mini", "gpt41mini": "GPT 4.1 mini"}


def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1))


def load(dirs, arm):
    rows = []  # one per suite
    for d in dirs:
        for f in sorted(glob.glob(f"{ROOT}/results/{d}/*.json")):
            j = json.load(open(f))
            for sid, s in j["suites"].items():
                m, cond, k = sid.split("_")
                if arm == "od": cond = cond.replace("plain", "od")
                imported = not all(t.startswith("::") for t in s["tests"])
                first = set(s["first_run_failed"])
                kept = {t: v for t, v in s["tests"].items() if t not in first} if imported else {}
                fl = sorted(t for t, v in kept.items() if v["kind"] == "flaky")
                removed_fl = sum(1 for t, v in s["tests"].items() if t in first and v["kind"] == "flaky")
                so = sum(1 for t in fl if kept[t]["fails_file_order"] == 0)
                rows.append(dict(project=j["project"], model=m, cond=cond, sample=k, sid=sid, imported=imported,
                                 generated=len(s["tests"]) if imported else 0, kept=len(kept), flaky=fl,
                                 removed_flaky=removed_fl, shuffle_only=so, runs=s["runs"]))
    return rows


def group(rows, model, cond, skip=()):
    r = [x for x in rows if x["model"] == model and x["cond"] == cond and x["project"] not in skip]
    imp = [x for x in r if x["imported"]]
    wf = [x for x in imp if x["flaky"]]
    ci = wilson(len(wf), len(imp))
    return dict(suites=len(r), imported=len(imp), kept=sum(x["kept"] for x in imp), suites_with_flaky=len(wf),
                pct=round(100 * len(wf) / len(imp), 1) if imp else 0, ci=ci,
                projects=sorted({x["project"] for x in wf}), flaky=sum(len(x["flaky"]) for x in imp),
                removed_flaky=sum(x["removed_flaky"] for x in imp), shuffle_only=sum(x["shuffle_only"] for x in imp),
                runs=sorted({x["runs"] for x in r}),
                by_project={p: [sum(1 for x in wf if x["project"] == p), sum(len(x["flaky"]) for x in wf if x["project"] == p)]
                            for p in sorted({x["project"] for x in wf})})


def main():
    held = load(["matched_heldout"], "heldout")
    od = load(["matched_od"], "od")
    g3 = load(["matched_guard3cl", "matched_guard3gpt"], "g3")
    out = {"heldout": {}, "od": {}, "guard3": {}}
    print("== held out (plain, M1 seed, M2 iso)")
    for m in MODELS:
        for cond in ("heldout", "heldoutseed", "heldoutiso"):
            for skip in ((), ("jmcm",)):
                g = group(held, m, cond, skip)
                key = f"{m}_{cond}" + ("_nojmcm" if skip else "")
                out["heldout"][key] = g
                print(f"{key:28s} suites {g['suites']} imp {g['imported']} kept {g['kept']:,} wf {g['suites_with_flaky']} "
                      f"({g['pct']}%, {g['ci'][0]}-{g['ci'][1]}) proj {len(g['projects'])} flaky {g['flaky']} "
                      f"removed {g['removed_flaky']} shuffle_only {g['shuffle_only']} runs {g['runs']}")
            print("   by project (plain):", out["heldout"][f"{m}_heldout"]["by_project"])
    print("== order dependent arm")
    for m in MODELS:
        for cond in ("od", "odseed", "odiso"):
            g = group(od, m, cond); out["od"][f"{m}_{cond}"] = g
            print(f"{m}_{cond:8s} suites {g['suites']} imp {g['imported']} kept {g['kept']:,} wf {g['suites_with_flaky']} "
                  f"flaky {g['flaky']} removed {g['removed_flaky']} so {g['shuffle_only']} {g['by_project']}")
    print("== guard v3 (samples 1-2 plain vs guard3)")
    for m in MODELS:
        sub = [x for x in held if x["model"] == m and x["cond"] == "heldout" and x["sample"] in ("1", "2")]
        for name, rs, cond in (("plain", sub, "heldout"), ("guard3", g3, "guard3")):
            g = group(rs, m, cond); out["guard3"][f"{m}_{name}"] = g
            print(f"{m}_{name:7s} imp {g['imported']} kept {g['kept']:,} wf {g['suites_with_flaky']} ({g['flaky']}) {g['by_project']}")
    json.dump(out, open(f"{ROOT}/results/report3_numbers.json", "w"), indent=1)


if __name__ == "__main__":
    main()
