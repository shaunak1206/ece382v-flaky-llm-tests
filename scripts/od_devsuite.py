#!/usr/bin/env python3
"""Order dependent arm inside the developer suite (Progress Report 3, goal 3).

For one order dependent arm project: set it up as matched_run.py does, then run the developer test file that holds
the known order dependent test together with one generated suite, in a new pytest-randomly order each run (seed and
PYTHONHASHSEED 9000+i, as the reproduction of this batch used). pytest-randomly shuffles the module order, so in about
half of the runs the generated file runs before the developer file. Configurations: the developer file alone
(baseline) and with each plain generated suite of the arm (<model>_plain_<k>). For every configuration it records,
per run, whether the generated module ran first, and per test its outcome, so we can ask:
  - does the developer order dependent test fail more often with a generated suite present,
  - does any other developer test fail with a generated suite present but never in the baseline (a generated test
    acting as a polluter of developer tests), and only when the generated module ran first,
  - do generated tests flip when run next to the developer tests.

  python scripts/od_devsuite.py <Project_Name> --runs 60 --out results/od_devsuite/<Project_Name>.json
"""
import argparse, collections, glob, json, os, shutil, sys, tempfile, time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruber_reproduce as gr

ROOT = gr.ROOT


def run(py, repo, targets, seed, timeout_s):
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    path = os.pathsep.join([repo] + ([f"{repo}/src"] if os.path.isdir(f"{repo}/src") else []))
    env = dict(os.environ, PYTHONPATH=path, PYTHONHASHSEED=str(seed))
    cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--timeout={timeout_s}", "-o", "python_files=test_*.py",
           "-p", "randomly", f"--randomly-seed={seed}", "--junitxml", xml] + targets
    rc, out = gr.sh(cmd, cwd=repo, timeout=900, env=env)
    order, res = [], {}
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            tid = f'{tc.get("classname")}::{tc.get("name")}'; st = "pass"
            for ch in tc:
                if ch.tag in ("failure", "error", "skipped"):
                    st = {"failure": "fail", "error": "error", "skipped": "skip"}[ch.tag]
            res[tid] = st; order.append(tid)
    except Exception:
        pass
    finally:
        os.unlink(xml)
    return res, order, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--work", default="/tmp/devsuite"); ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=60); ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--suites", default="*_plain_*"); ap.add_argument("--max-minutes", type=int, default=320)
    ap.add_argument("--alone", action="store_true", help="control: run each generated suite without the developer file, "
                    "same repo, seeds, and shuffling, so flips that need the developer tests can be told apart")
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
    gr.sh(["uv", "pip", "install", "-q", "-p", py, "pytest-randomly<3.16", "freezegun", "pytest-mock"])
    gr.fix_missing_modules(py, repo, r["Test_filename"], log)
    devfile, odtest = r["Test_filename"], r["Test_funcname"]
    res = {"project": a.project, "install_ok": ok, "dev_file": devfile, "od_test": odtest, "runs": a.runs, "configs": {}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sdirs = sorted({x for pat in a.suites.split(",") for x in glob.glob(f"{ROOT}/matched/{a.project}/{pat}/")})
    configs = ([] if a.alone else [None]) + [os.path.basename(x.rstrip("/")) for x in sdirs]
    res["alone"] = a.alone
    for sid in configs:
        targets = [] if a.alone else [devfile]
        if sid:
            dst = f"{repo}/zz_{sid}"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
            for f in glob.glob(f"{ROOT}/matched/{a.project}/{sid}/*.py"):
                if os.path.basename(f) not in ("__init__.py", "conftest.py"): shutil.copy(f, dst)
            targets.append(f"zz_{sid}")
        hist = collections.defaultdict(collections.Counter)
        gen_first = []  # per run: did a generated test run before the first developer test
        fails_when = collections.defaultdict(collections.Counter)  # dev test -> {gen_first/dev_first: failures}
        for i in range(a.runs):
            out_res, order, tail = run(py, repo, targets, 9000 + i, a.timeout)
            if tail.startswith("TIMEOUT"):  # a generated test that never ends (same rule as matched_run.py)
                res.setdefault("nonterminating", []).append(sid); break
            gf = bool(order) and order[0].startswith("zz_")
            gen_first.append(gf)
            for t, s in out_res.items():
                hist[t][s] += 1
                if s in ("fail", "error") and not t.startswith("zz_"):
                    fails_when[t]["gen_first" if gf else "dev_first"] += 1
            if time.time() - t0 > a.max_minutes * 60: break
        tests = {t: dict(c) for t, c in hist.items()}
        od = {k: sum(v.get(k, 0) for t, v in tests.items() if not t.startswith("zz_") and t.split("::")[-1].split("[")[0] == odtest)
              for k in ("pass", "fail", "error")}
        res["configs"][sid or "baseline"] = {
            "runs_done": len(gen_first), "gen_first_runs": sum(gen_first), "od_test": od,
            "dev_failing": {t: dict(fails_when[t]) for t in fails_when},
            "gen_flaky": sorted(t for t, c in tests.items() if t.startswith("zz_") and c.get("pass") and (c.get("fail", 0) + c.get("error", 0))),
            "tests": tests}
        json.dump(res, open(a.out, "w"), indent=1)
        b = res["configs"].get("baseline", {}).get("dev_failing", {}) if not a.alone else {}
        new = [t for t in fails_when if t not in b]
        print(f"[{a.project}/{sid or 'baseline'}] runs={len(gen_first)} od={od} new_dev_failing={new[:3]} "
              f"gen_flaky={len(res['configs'][sid or 'baseline']['gen_flaky'])}", flush=True)
        if time.time() - t0 > a.max_minutes * 60:
            res["stopped_early"] = True; break
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
