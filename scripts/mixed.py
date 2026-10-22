#!/usr/bin/env python3
"""Mixed suites: run the generated tests inside the library's own test suite, the way a developer
would actually add them, so either side can pollute the other (proposal feedback question 3).

The generated files (only the tests a pipeline would keep) are copied into a zz_gen_<cond> folder
inside the subject's test directory, and pytest runs the human and generated tests in ONE session
in random order (pytest-randomly), with a new hash seed each run. With --human-only the same thing
runs without the generated tests, as a same machine baseline.

usage: python scripts/mixed.py <cond|human> <proj> --runs 30 --out results/mixed/<cond>/<proj>.json
"""
import argparse, glob, json, os, shutil, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# proj -> (subject dir, human test target, extra pytest args, extra env), same as the human baseline runs
HUMAN = {
 "cachetools": ("cachetools", "tests", [], {}),
 "schedule": ("schedule", "test_schedule.py", [], {}),
 "tinydb": ("tinydb", "tests", ["-o", "addopts="], {}),
 "humanize": ("humanize", "tests", ["--ignore=tests/test_benchmarks.py"], {}),
 "toolz": ("toolz", "toolz", [], {}),
 "tabulate": ("python-tabulate", "test", [], {}),
 "tenacity": ("tenacity", "tests", ["-o", "addopts=", "--timeout=120"], {}),
 "filelock": ("filelock", "tests", ["-o", "addopts=", "--timeout=120"], {"PYTHONPATH": "."}),
 "boltons": ("boltons", "tests", ["-o", "addopts=", "--timeout=120"], {}),
 "more_itertools": ("more-itertools", "tests", ["-o", "addopts=", "--timeout=120"], {}),
 "arrow": ("arrow", "tests", ["-o", "addopts=", "--timeout=120"], {}),
 "pybreaker": ("pybreaker", "tests", ["-o", "addopts=", "--timeout=120"], {}),
}

def kept_deselects(cond, proj, gendir_rel):
    """Node ids of tests the one run filter rejected, rewritten to their copied location."""
    f = f"{ROOT}/results/filter/{cond}/{proj}.json"
    if not os.path.exists(f): return []
    j = json.load(open(f))
    out = []
    for tid in j["rejected_ids"]:
        cls, name = tid.split("::", 1)
        parts = cls.split(".")
        # classname is <cond>.<proj>.<module>[.<Class>]
        rest = parts[2:] if parts[:2] == [cond, proj] else parts
        mod, klass = rest[0], rest[1:]
        out.append("::".join([f"{gendir_rel}/{mod}.py"] + klass + [name]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond"); ap.add_argument("proj")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--mode", default="both")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sub, htarget, extra, env_extra = HUMAN[a.proj]
    sdir = f"{ROOT}/subjects/{sub}"
    targets = [htarget]
    gendir = None
    deselect_file = None
    if a.cond != "human":
        base = htarget if os.path.isdir(f"{sdir}/{htarget}") else "."
        gendir_rel = f"{base}/zz_gen_{a.cond}".lstrip("./") if base != "." else f"zz_gen_{a.cond}"
        gendir = f"{sdir}/{gendir_rel}"
        shutil.rmtree(gendir, ignore_errors=True)
        os.makedirs(gendir)
        for f in glob.glob(f"{ROOT}/{a.cond}/{a.proj}/*.py"):
            if os.path.basename(f) not in ("__init__.py", "conftest.py"):
                shutil.copy(f, gendir)
        targets.append(gendir_rel)
        des = kept_deselects(a.cond, a.proj, gendir_rel)
        deselect_file = f"{gendir}.deselect"
        open(deselect_file, "w").write("\n".join(des))
    env = dict(os.environ, **env_extra)
    cmd = [sys.executable, f"{ROOT}/scripts/rerun.py", "--name", f"mixed-{a.cond}-{a.proj}", "--cwd", sdir,
           "--target", ",".join(targets), "--runs", str(a.runs), "--mode", a.mode, "--out", a.out]
    if deselect_file: cmd += ["--deselect-file", deselect_file]
    cmd += ["--", "--continue-on-collection-errors", "-p", "no:cacheprovider"] + extra
    if "--timeout=120" not in extra: cmd.append("--timeout=60")
    try:
        rc = subprocess.call(cmd, env=env)
    finally:
        if gendir:
            shutil.rmtree(gendir, ignore_errors=True)
            if deselect_file and os.path.exists(deselect_file): os.remove(deselect_file)
    sys.exit(rc)

if __name__ == "__main__":
    main()
