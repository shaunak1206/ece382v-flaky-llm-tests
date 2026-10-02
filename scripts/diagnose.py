#!/usr/bin/env python3
"""Targeted reruns for each flaky test in results/summary.json to attribute the perturbation.

For every flaky test it measures the failure rate in five configurations:
  iso_plain     the test alone, fixed hash seed, local timezone            (intrinsic: timing, random, clock)
  iso_hash      the test alone, varying PYTHONHASHSEED                     (set / dict hash order)
  iso_tz        the test alone, varying TZ                                 (timezone)
  iso_stress    the test alone, CPU burners on every core                  (timing under load)
  suite_order   whole suite, random test order each run                    (order dependence)
Writes results/diagnosis.json
"""
import json, os, subprocess, sys
R = "/Users/shaunkapoor/ece382v-flaky-llm-tests"
PY = f"{R}/.venv/bin/python"
SUITE_DIR = {"llm_tests": R + "/llm_tests", "gen_oneshot_sonnet": R + "/gen_oneshot_sonnet", "gen_oneshot_haiku": R + "/gen_oneshot_haiku"}
N = int(os.environ.get("DIAG_RUNS", "50"))


def nodeid(suite, junit_id):
    """junit 'pkg.dirs.module.Class::test' -> '<suite>/module.py::Class::test' (finds the file part)."""
    mod, name = junit_id.split("::", 1)
    parts = mod.split(".")
    for i in range(len(parts)):
        if os.path.exists(f"{suite}/{parts[i]}.py"):
            return f"{suite}/{parts[i]}.py::" + "::".join(parts[i + 1:] + [name])
    return f"{suite}/{parts[-1]}.py::{name}"


def rate(target, mode, cwd, name):
    out = f"{R}/results/diag/{name}.{mode}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([PY, f"{R}/scripts/rerun.py", "--name", name, "--cwd", cwd, "--target", target, "--runs", str(N),
                    "--mode", mode, "--out", out, "--", "--timeout=60", "--continue-on-collection-errors"],
                   capture_output=True, text=True)
    return json.load(open(out))


def main():
    S = json.load(open(f"{R}/results/summary.json"))
    diag = []
    for i, f in enumerate(S["flaky"]):
        if f["cond"] == "human":
            continue  # human flaky tests are diagnosed by hand
        suite = f"{SUITE_DIR[f['cond']]}/{f['proj']}"
        nid = nodeid(suite, f["test"])
        work = f"{R}/results/work/diag/{i}"
        os.makedirs(work, exist_ok=True)
        row = {"cond": f["cond"], "proj": f["proj"], "test": f["test"]}
        for label, target, mode in [("iso_plain", nid, "plain"), ("iso_hash", nid, "hashseed"), ("iso_tz", nid, "tz"),
                                    ("iso_stress", nid, "stress"), ("suite_order", suite, "order")]:
            d = rate(target, mode, work, f"d{i}_{label}")
            t = d["tests"].get(f["test"], {"counts": {}})
            c = t["counts"]
            fails = c.get("fail", 0) + c.get("error", 0)
            row[label] = f"{fails}/{fails + c.get('pass', 0)}"
        diag.append(row)
        print(row, flush=True)
    json.dump(diag, open(f"{R}/results/diagnosis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
