#!/usr/bin/env python3
"""Targeted reruns that separate order dependent from non order dependent flaky generated tests (E3).

For one matched study project: set it up the same way as matched_run.py, then for every generated test that
was flaky in results/matched*/<project>.json
  alone:  run the test by itself in fresh processes, 20 times with a new hash seed each time and 10 times with
          PYTHONHASHSEED=0. A test that fails alone is not order dependent; one that fails only with varying
          hash seeds depends on the salted builtin hash().
  pairs:  if it never fails alone, run every other test of its suite and then the test (polluter, victim),
          3 times each with PYTHONHASHSEED=0, the pairwise search of iDFlakies and iFixFlakies. The tests whose
          run before it makes it fail are its polluters.

  python scripts/od_diagnose.py <Project_Name> --out results/od_diagnose/<Project_Name>.json

Output JSON: project, tests: [{suite, test, alone_varied_fail, alone_fixed_fail, pairs_tried, polluters, verdict}]
verdict is one of od (passes alone, fails after a polluter), nod-hash (fails alone only with varied hash seeds),
nod (fails alone with a fixed hash seed), unknown (passes alone and after every single other test).
"""
import argparse, glob, json, os, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruber_reproduce as gr

ROOT = gr.ROOT


def node_id(tid):
    # junit id "zz_sid.test_gen::name" or "zz_sid.test_gen.Class::name" -> pytest node id
    cls, name = tid.split("::", 1)
    parts = cls.split(".")
    path = "/".join(parts[:2]) + ".py"
    return "::".join([path] + parts[2:] + [name])


def run(py, repo, ids, seed, timeout_s=120):
    path = os.pathsep.join([repo] + ([f"{repo}/src"] if os.path.isdir(f"{repo}/src") else []))
    env = dict(os.environ, PYTHONPATH=path, PYTHONHASHSEED=str(seed))
    cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", f"--timeout={timeout_s}", "-rA"]
    rc, out = gr.sh(cmd + ids, cwd=repo, timeout=600, env=env)
    last = ids[-1].split("::")[-1]
    for line in out.splitlines():
        if line.startswith(("PASSED", "FAILED", "ERROR")) and line.split(" ", 1)[1].split(" - ")[0].endswith("::" + last):
            return line.split(" ", 1)[0]
    return "NONE"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--work", default="/tmp/od_work"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = gr.row_for(a.project); t0 = time.time()
    d = f"{a.work}/{a.project}"; repo = f"{d}/repo"; venv = f"{d}/venv"
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    gr.sh(["git", "init", "-q", repo]); gr.sh(["git", "-C", repo, "remote", "add", "origin", r["Project_URL"]])
    gr.sh(["git", "-C", repo, "fetch", "-q", "--depth", "1", "origin", r["Project_Hash"]], env=env)
    gr.sh(["git", "-C", repo, "checkout", "-q", "FETCH_HEAD"])
    log = []
    ok, py = gr.install(repo, venv, log)
    gr.sh(["uv", "pip", "install", "-q", "-p", py, "pytest-randomly<3.16", "freezegun", "pytest-mock", "pytest-timeout"])
    gr.fix_missing_modules(py, repo, r["Test_filename"], log)
    targets = []  # (sid, test id, all test ids of the suite)
    for f in glob.glob(f"{ROOT}/results/matched*/{a.project}.json"):
        for sid, s in json.load(open(f))["suites"].items():
            for t, v in s["tests"].items():
                if v["kind"] == "flaky" and (sid, t) not in {(x, y) for x, y, _ in targets}:
                    targets.append((sid, t, sorted(s["tests"])))
    out = {"project": a.project, "install_ok": ok, "tests": []}
    for sid in sorted({s for s, _, _ in targets}):
        model, prompt = sid.split("_")[:2]
        dst = f"{repo}/zz_{sid}"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
        for f in glob.glob(f"{ROOT}/matched/{a.project}/{sid}/*.py"):
            if os.path.basename(f) == "__init__.py" or (os.path.basename(f) == "conftest.py" and prompt not in ("plainseed", "plainiso")): continue
            shutil.copy(f, dst)
    for sid, t, suite in targets:
        v = node_id(t)
        varied = sum(run(py, repo, [v], 9000 + i) != "PASSED" for i in range(20))
        fixed = sum(run(py, repo, [v], 0) != "PASSED" for _ in range(10))
        polluters, tried = [], 0
        if not varied and not fixed:
            for other in suite:
                if other == t: continue
                tried += 1
                if any(run(py, repo, [node_id(other), v], 0) != "PASSED" for _ in range(3)):
                    polluters.append(other.split("::")[-1])
        verdict = ("nod" if fixed else "nod-hash" if varied else "od" if polluters else "unknown")
        out["tests"].append({"suite": sid, "test": t.split("::")[-1], "alone_varied_fail": varied, "alone_fixed_fail": fixed,
                             "pairs_tried": tried, "polluters": polluters, "verdict": verdict})
        print(f"[{a.project}/{sid}] {t.split('::')[-1]} {verdict} varied={varied}/20 fixed={fixed}/10 polluters={polluters}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
