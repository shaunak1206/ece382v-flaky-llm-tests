#!/usr/bin/env python3
"""Summarize results/human/*.json and results/llm/*.json into one table + flaky test listing."""
import json, glob, os, collections, re

CATS = [  # crude keyword-based root-cause labels (to be refined manually)
    ("time",        r"time|second|minute|hour|ago|from now|expire|ttl|naturaltime|naturalday|sleep|monotonic"),
    ("order/set",   r"set\(|frozenset|order|sorted|!= \[|== \[|\{.*\} == \{|dict|keys"),
    ("random",      r"random|sample|choice|RRCache"),
    ("filesystem",  r"file|path|exist|directory|tmp|json"),
    ("global state",r"activate|locale|default|schedule.jobs|clear"),
]
def label(msgs):
    text = " ".join(msgs)
    return ",".join(c for c, pat in CATS if re.search(pat, text, re.I)) or "unknown"

rows = []
for kind in ("human", "llm"):
    for f in sorted(glob.glob(f"results/{kind}/*.json")):
        d = json.load(open(f))
        proj = os.path.basename(f).split(".")[0]
        rows.append((kind, proj, d["mode"], d["runs"], d["num_tests"] or 0, d["num_flaky"],
                     d["num_always_fail"], d["runs_with_failure"], d["avg_run_seconds"], d))
print(f"{'suite':6} {'project':16} {'mode':5} {'runs':>4} {'tests':>5} {'flaky':>5} {'flaky%':>6} {'alwaysF':>7} {'badRuns':>7} {'sec/run':>7}")
tot = collections.Counter()
for kind, proj, mode, runs, n, fl, af, br, sec, d in rows:
    pct = 100*fl/n if n else 0
    print(f"{kind:6} {proj:16} {mode:5} {runs:4d} {n:5d} {fl:5d} {pct:5.1f}% {af:7d} {br:7d} {sec:7.1f}")
    tot[kind+"_tests"] += n; tot[kind+"_flaky"] += fl
for kind in ("human", "llm"):
    n, fl = tot[kind+"_tests"], tot[kind+"_flaky"]
    print(f"TOTAL {kind:6} tests={n} flaky={fl} ({100*fl/n if n else 0:.2f}%)")
print("\nFlaky tests (with first failure message and crude category):")
for kind, proj, mode, runs, n, fl, af, br, sec, d in rows:
    for t in d["flaky_tests"]:
        info = d["tests"][t]
        print(f"  [{kind}/{proj}] {t}  {info['counts']}  cat={label(info['messages'])}")
        for m in info["messages"][:1]:
            print(f"      msg: {m[:160]}")
