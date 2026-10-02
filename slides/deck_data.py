#!/usr/bin/env python3
"""Collect the numbers the lightning-talk deck shows into slides/deck_data.json."""
import json
R = "/Users/shaunkapoor/ece382v-flaky-llm-tests/results"
S = json.load(open(f"{R}/summary.json"))
SM = json.load(open(f"{R}/smells.json"))
LAB = {"human": "Human", "llm_tests": "Agentic Sonnet", "gen_oneshot_sonnet": "One shot Sonnet", "gen_oneshot_haiku": "One shot Haiku"}

executions = 0
tests_total = 0
for k, v in S["suites"].items():
    t = v["tests"]
    tests_total += t
    executions += t * sum(v["runs_per_mode"].values())

smell = {}
for cond, lab in LAB.items():
    ss = [v for k, v in SM.items() if k.startswith(cond + "/")]
    n = sum(s["n_tests"] for s in ss)
    smell[lab] = round(100 * sum(s["any"] for s in ss) / n, 1) if n else 0

pooled = {LAB[c]: v for c, v in S["pooled"].items()}
flaky_total = sum(1 for f in S["flaky"])
llm_flaky = sum(1 for f in S["flaky"] if f["cond"] != "human")
human_flaky = flaky_total - llm_flaky
filt = {LAB[c]: v for c, v in S["filter"].items()}
out = {"suites": len(S["suites"]), "tests_total": tests_total, "executions": executions,
       "flaky_total": flaky_total, "llm_flaky": llm_flaky, "human_flaky": human_flaky,
       "pooled": pooled, "smell_any_pct": smell, "filter": filt}
json.dump(out, open("/Users/shaunkapoor/ece382v-flaky-llm-tests/slides/deck_data.json", "w"), indent=1)
print(json.dumps(out, indent=1))
