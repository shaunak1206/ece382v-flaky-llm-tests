#!/usr/bin/env python3
"""Pairwise polluter search for the developer file runs (Progress Report 3).

For one order dependent arm project, using results/od_devsuite/<project>.json and its --alone control:
  - developer victims: a developer test that fails only next to a generated suite; try each generated test of
    that suite before it (generated test, developer test), 3 times with seed 0.
  - generated victims: a generated test that flips in the combined runs but never alone with the same seeds; try
    each developer test of the file before it (developer test, generated test), 3 times.
A polluter is confirmed when the victim then fails in all 3 tries and passes alone 3 of 3.
Writes results/devsuite_polluters/<project>.json.

  python scripts/devsuite_polluters.py <Project_Name> --out results/devsuite_polluters/<Project_Name>.json
"""
import argparse, glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruber_reproduce as gr
from od_diagnose import run

ROOT = gr.ROOT


def node(tid, devfile):
    cls, name = tid.split("::", 1)
    if cls.startswith("zz_"):
        parts = cls.split("."); return "/".join(parts[:2]) + ".py::" + "::".join(parts[2:] + [name])
    mod = devfile[:-3].replace("/", ".")
    rest = cls[len(mod) + 1:].split(".") if cls.startswith(mod + ".") else []
    return "::".join([devfile] + [x for x in rest if x] + [name])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--work", default="/tmp/dvp"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    comb = json.load(open(f"{ROOT}/results/od_devsuite/{a.project}.json"))
    alone_f = f"{ROOT}/results/od_devsuite_alone/{a.project}.json"
    alone = json.load(open(alone_f))["configs"] if os.path.exists(alone_f) else {}
    r = gr.row_for(a.project); d = f"{a.work}/{a.project}"; repo = f"{d}/repo"; venv = f"{d}/venv"
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    gr.sh(["git", "init", "-q", repo]); gr.sh(["git", "-C", repo, "remote", "add", "origin", r["Project_URL"]])
    gr.sh(["git", "-C", repo, "fetch", "-q", "--depth", "1", "origin", r["Project_Hash"]], env=env)
    gr.sh(["git", "-C", repo, "checkout", "-q", "FETCH_HEAD"])
    ok, py = gr.install(repo, venv, [])
    gr.sh(["uv", "pip", "install", "-q", "-p", py, "pytest-randomly<3.16", "freezegun", "pytest-mock", "pytest-timeout"])
    devfile = comb["dev_file"]; base = comb["configs"].get("baseline", {})
    out = {"project": a.project, "victims": []}
    for sid, c in comb["configs"].items():
        if sid == "baseline" or not c.get("runs_done"): continue
        dst = f"{repo}/zz_{sid}"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
        for f in glob.glob(f"{ROOT}/matched/{a.project}/{sid}/*.py"):
            if os.path.basename(f) not in ("__init__.py", "conftest.py"): shutil.copy(f, dst)
        gen_tests = [t for t in c["tests"] if t.startswith("zz_")]
        dev_tests = [t for t in c["tests"] if not t.startswith("zz_") and not t.startswith("::")]
        jobs = []
        for t in c.get("dev_failing", {}):
            if t not in base.get("dev_failing", {}) and not t.startswith("::"): jobs.append(("developer victim", t, gen_tests))
        al = {x.split("::")[-1] for x in (alone.get(sid) or {}).get("gen_flaky", [])} if sid in alone else None
        for t in c.get("gen_flaky", []):
            if al is None or t.split("::")[-1] not in al: jobs.append(("generated victim", t, dev_tests))
        for kind, v, cands in jobs:
            vn = node(v, devfile)
            rec = {"suite": sid, "kind": kind, "victim": v, "alone_pass": sum(run(py, repo, [vn], 0) == "PASSED" for _ in range(3)),
                   "control": "same-seed control" if al is not None else "no control", "polluters": []}
            for p in cands:
                pn = node(p, devfile)
                if all(run(py, repo, [pn, vn], 0) != "PASSED" for _ in range(3)): rec["polluters"].append(p.split("::")[-1])
            out["victims"].append(rec)
            json.dump(out, open(a.out, "w"), indent=1)
            print(f"[{a.project}/{sid}] {kind} {v.split('::')[-1]} polluters={rec['polluters'][:3]}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
