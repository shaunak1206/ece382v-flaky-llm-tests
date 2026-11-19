#!/usr/bin/env python3
"""Per model, arm and fixture: of the kept tests that never failed in 200 plain runs, how many fail every run with
the fixture (broken) and how many flip with it (made flaky). Reads results/matched_heldout and results/matched_od.
Writes results/fixture_effects.json.

  python scripts/fixture_effects.py
"""
import collections, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    out = {}
    for d, arm, base in (("matched_heldout", "heldout", "heldout"), ("matched_od", "od", "plain")):
        cell = collections.defaultdict(lambda: {"stable_kept": 0, "broken": [], "made_flaky": []})
        for f in sorted(glob.glob(f"{ROOT}/results/{d}/*.json")):
            j = json.load(open(f)); S = j["suites"]
            for sid, s in S.items():
                m, cond, k = sid.split("_")
                if cond != base or all(t.startswith("::") for t in s["tests"]): continue
                first = set(s["first_run_failed"])
                stable = {t.split("::")[-1]: t for t, v in s["tests"].items() if t not in first and v["kind"] == "stable"}
                for fx in ("seed", "iso"):
                    o = S.get(f"{m}_{base}{fx}_{k}")
                    if not o: continue
                    c = cell[f"{m}_{fx}"]; c["stable_kept"] += len(stable)
                    by = {t.split("::")[-1]: v for t, v in o["tests"].items()}
                    for name in stable:
                        v = by.get(name)
                        if not v: continue
                        if v["kind"] == "always-fail": c["broken"].append(f"{j['project']}/{sid}::{name}")
                        elif v["kind"] == "flaky": c["made_flaky"].append(f"{j['project']}/{sid}::{name}")
        out[arm] = cell
    json.dump(out, open(f"{ROOT}/results/fixture_effects.json", "w"), indent=1)
    for arm, cells in out.items():
        for k, c in sorted(cells.items()):
            print(f"{arm:8s} {k:16s} stable kept {c['stable_kept']:6d} broken {len(c['broken'])} made flaky {len(c['made_flaky'])} {c['made_flaky'][:3]} {c['broken'][:2]}")


if __name__ == "__main__":
    main()
