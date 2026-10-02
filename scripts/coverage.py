#!/usr/bin/env python3
"""Line coverage of the library under test for each suite (human, agentic, one-shot accepted-only)."""
import json, os, subprocess, sys
R = "/Users/shaunkapoor/ece382v-flaky-llm-tests"
PY = f"{R}/.venv/bin/python"
SUBJ = {  # proj: (subject dir, human target, coverage package, extra args, module filter)
 "cachetools": ("cachetools", "tests", "cachetools", [], None),
 "schedule": ("schedule", "test_schedule.py", "schedule", [], None),
 "tinydb": ("tinydb", "tests", "tinydb", [], None),
 "humanize": ("humanize", "tests", "humanize", ["--ignore=tests/test_benchmarks.py"], None),
 "toolz": ("toolz", "toolz", "toolz", [], None),
 "tabulate": ("python-tabulate", "test", "tabulate", [], None),
 "tenacity": ("tenacity", "tests", "tenacity", [], None),
 "filelock": ("filelock", "tests", "filelock", [], None),
 "boltons": ("boltons", "tests", "boltons", [], ["cacheutils", "iterutils", "timeutils", "queueutils", "fileutils", "dictutils", "setutils", "strutils"]),
 "more_itertools": ("more-itertools", "tests", "more_itertools", [], None),
 "arrow": ("arrow", "tests", "arrow", [], None),
 "pybreaker": ("pybreaker", "tests", "pybreaker", [], None),
}
def run(cwd, target, pkg, extra, outjson, env_extra=None):
    env = dict(os.environ, PYTHONHASHSEED="0", **(env_extra or {}))
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:randomly", "-p", "no:cacheprovider", "-o", "addopts=",
           "--continue-on-collection-errors", "--timeout=120", f"--cov={pkg}", f"--cov-report=json:{outjson}",
           "--cov-config=/dev/null", target] + extra
    subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
def pct(outjson, modfilter):
    if not os.path.exists(outjson): return None
    d = json.load(open(outjson)); cov = miss = 0
    for f, v in d["files"].items():
        if "/tests/" in f or "/test" in os.path.basename(f): continue
        if modfilter and not any(f.endswith(f"/{m}.py") for m in modfilter): continue
        cov += v["summary"]["covered_lines"]; miss += v["summary"]["missing_lines"]
    return round(100 * cov / max(cov + miss, 1), 1)
def nodeid(suite, junit_id):
    """junit 'pkg.dirs.module.Class::test' -> '<suite>/module.py::Class::test' (finds the file part)."""
    mod, name = junit_id.split("::", 1)
    parts = mod.split(".")
    for i in range(len(parts)):
        if os.path.exists(f"{suite}/{parts[i]}.py"):
            return f"{suite}/{parts[i]}.py::" + "::".join(parts[i + 1:] + [name])
    return f"{suite}/{parts[-1]}.py::{name}"
res = {}
os.makedirs(f"{R}/results/coverage", exist_ok=True)
for proj, (sd, tgt, pkg, extra, mf) in SUBJ.items():
    row = {}
    o = f"{R}/results/coverage/human_{proj}.json"
    run(f"{R}/subjects/{sd}", tgt, pkg, extra, o, {"PYTHONPATH": "."} if proj == "filelock" else None); row["human"] = pct(o, mf)
    for cond in ("llm_tests", "gen_oneshot_sonnet", "gen_oneshot_haiku"):
        suite = f"{R}/{cond}/{proj}"
        if not os.path.isdir(suite): continue
        des = []
        fj = f"{R}/results/filter/{cond}/{proj}.json"
        if os.path.exists(fj):
            for t in json.load(open(fj))["rejected_ids"]:
                des += ["--deselect", nodeid(suite, t)]
        work = f"{R}/results/work/{cond}/{proj}/cov"; os.makedirs(work, exist_ok=True)
        o = f"{R}/results/coverage/{cond}_{proj}.json"; run(work, suite, pkg, des, o); row[cond] = pct(o, mf)
    res[proj] = row; print(proj, row, flush=True)
json.dump(res, open(f"{R}/results/coverage/summary.json", "w"), indent=1)
