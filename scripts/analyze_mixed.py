#!/usr/bin/env python3
"""Mixed suites (generated tests inside the human suite, one pytest session, random order, GitHub Actions).
Compares each mixed run with the human suite run alone on the same machine type and reports
flaky tests split by origin (human or generated), and human tests that only flip when mixed (pollution)."""
import json, glob, os, collections
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results/mixed"
LAB = {"llm_tests": "AG-S", "gen_oneshot_haiku": "OS-H", "gen_tp_sonnet": "TP-S", "gen_plain_sonnet": "PL-S"}
def load(f):
    j = json.load(open(f)); return j
human = {os.path.basename(f)[len("human__"):-5]: load(f) for f in glob.glob(f"{R}/human__*.json")}
rows = []; out = {"human_alone": {}, "mixed": {}}
for p, j in sorted(human.items()):
    out["human_alone"][p] = {"tests": j["num_tests"], "flaky": j["flaky_tests"], "always_fail": j["num_always_fail"]}
for f in sorted(glob.glob(f"{R}/*__*.json")):
    cond, proj = os.path.basename(f)[:-5].split("__")
    if cond == "human": continue
    j = load(f)
    gen = [t for t in j["tests"] if "zz_gen" in t]; hum = [t for t in j["tests"] if "zz_gen" not in t]
    fl_gen = [t for t in j["flaky_tests"] if "zz_gen" in t]; fl_hum = [t for t in j["flaky_tests"] if "zz_gen" not in t]
    base = set(human.get(proj, {}).get("flaky_tests", []))
    new_hum = [t for t in fl_hum if t not in base]
    # human tests that pass alone but always fail once generated tests share the session
    halone = human.get(proj, {}).get("tests", {})
    broke = [t for t in hum if j["tests"][t]["kind"] == "always-fail" and halone.get(t, {}).get("kind") == "stable"]
    out["mixed"][f"{cond}/{proj}"] = {"gen_tests": len(gen), "human_tests": len(hum), "flaky_generated": fl_gen,
                                      "flaky_human": fl_hum, "human_flaky_only_when_mixed": new_hum,
                                      "human_broken_by_mix": broke, "runs": j["runs"]}
agg = collections.defaultdict(lambda: collections.Counter())
for k, v in out["mixed"].items():
    c = k.split("/")[0]; a = agg[c]
    a["suites"] += 1; a["gen_tests"] += v["gen_tests"]; a["flaky_generated"] += len(v["flaky_generated"])
    a["flaky_human"] += len(v["flaky_human"]); a["human_flaky_only_when_mixed"] += len(v["human_flaky_only_when_mixed"])
    a["human_broken_by_mix"] += len(v["human_broken_by_mix"])
print("| Condition mixed in | Suites | Generated tests | Flaky generated | Flaky human | Human flaky only when mixed | Human tests broken by the mix |")
print("|---|---|---|---|---|---|---|")
for c in LAB:
    a = agg[c]
    if a["suites"]:
        print(f"| {LAB[c]} | {a['suites']} | {a['gen_tests']} | {a['flaky_generated']} | {a['flaky_human']} | {a['human_flaky_only_when_mixed']} | {a['human_broken_by_mix']} |")
print("\nHuman suites alone on the same runners: flaky", sum(len(v["flaky"]) for v in out["human_alone"].values()),
      {p: v["flaky"] for p, v in out["human_alone"].items() if v["flaky"]})
for k, v in out["mixed"].items():
    for key in ("flaky_generated", "human_flaky_only_when_mixed", "human_broken_by_mix"):
        for t in v[key]: print(f"  {k} {key}: {t}")
json.dump(out, open(f"{R}/../mixed_summary.json", "w"), indent=1)
