#!/usr/bin/env python3
"""Summarize results/od_devsuite (scripts/od_devsuite.py): per project and configuration, the developer order
dependent test's failure rate alone and with each generated suite, developer tests that fail only with a generated
suite present (and whether in runs where the generated module ran first), and generated tests that flip next to the
developer tests. Writes results/od_devsuite_summary.json.

  python scripts/od_devsuite_summary.py
"""
import collections, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    out = {"projects": {}, "polluted": [], "totals": collections.Counter()}
    for f in sorted(glob.glob(f"{ROOT}/results/od_devsuite/*.json")):
        j = json.load(open(f)); p = j["project"]; C = j["configs"]
        base = C.get("baseline")
        if not base or not base["runs_done"]: out["projects"][p] = {"usable": False}; continue
        def rate(c):
            o = c["od_test"]; n = o["pass"] + o["fail"] + o["error"]
            return round((o["fail"] + o["error"]) / n, 3) if n else None
        row = {"usable": True, "baseline_od_fail_rate": rate(base), "configs": {}, "nonterminating": j.get("nonterminating", [])}
        bfail = set(base["dev_failing"])
        for sid, c in C.items():
            if sid == "baseline" or not c["runs_done"]: continue
            new = {t: v for t, v in c["dev_failing"].items() if t not in bfail}
            row["configs"][sid] = {"runs": c["runs_done"], "gen_first_runs": c["gen_first_runs"], "od_fail_rate": rate(c),
                                   "new_dev_failing": new, "gen_flaky": len(c["gen_flaky"])}
            t = out["totals"]; t["configs"] += 1; t["runs"] += c["runs_done"]; t["gen_first_runs"] += c["gen_first_runs"]
            t["configs_with_new_dev_failing"] += bool(new); t["configs_with_gen_flaky"] += bool(c["gen_flaky"])
            for test, w in new.items():
                out["polluted"].append({"project": p, "suite": sid, "dev_test": test, **w})
        out["projects"][p] = row
    out["totals"] = dict(out["totals"])
    json.dump(out, open(f"{ROOT}/results/od_devsuite_summary.json", "w"), indent=1)
    print(json.dumps(out["totals"]))
    for p, r in out["projects"].items():
        if not r.get("usable"): print(p, "not usable"); continue
        rates = [c["od_fail_rate"] for c in r["configs"].values() if c["od_fail_rate"] is not None]
        print(f"{p:28s} base {r['baseline_od_fail_rate']} with-suites min {min(rates) if rates else None} max {max(rates) if rates else None} "
              f"configs {len(r['configs'])} newdev {sum(bool(c['new_dev_failing']) for c in r['configs'].values())} "
              f"genflaky {sum(c['gen_flaky'] for c in r['configs'].values())} nonterm {r['nonterminating']}")
    for x in out["polluted"]: print(x)


if __name__ == "__main__":
    main()
