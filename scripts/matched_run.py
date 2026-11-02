#!/usr/bin/env python3
"""Run LLM generated suites that target the same code as a known flaky human test (matched study).

For one Gruber et al. project: check out the commit, install it the same way as gruber_reproduce.py,
then for every generated suite in matched/<project>/<suite_id>/ copy its files into the project
(zz_<suite_id>/) and run the whole suite many times in fresh processes, half in file order and half in
random order (pytest-randomly) with a new hash seed. The known flaky human test is rerun too, as a
same machine control.

  python scripts/matched_run.py <Project_Name> --runs 150 --out results/matched/<Project_Name>.json

Output JSON fields
  project, sha, category (Gruber label of the human flaky test), install_ok,
  human: {runs, passed, failed, errors, reproduced, messages}
  suites: {suite_id: {model, prompt, sample, n_tests, first_run_failed (ids failing in run 1, what a one run
           filter would reject), tests: {test_id: {pass, fail, error, skip, kind}}, flaky (ids),
           always_fail (ids), runs}}
  kind is flaky (passed and failed), always-fail, stable, or always-skip.
"""
import argparse, collections, glob, json, os, random, shutil, sys, tempfile, time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruber_reproduce as gr

ROOT = gr.ROOT


def run_suite(py, repo, target_dir, timeout_s, seed, randomize):
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    # src layout projects that are not installable (fuzzylogic) put src/ on sys.path in their own tests
    path = os.pathsep.join([repo] + ([f"{repo}/src"] if os.path.isdir(f"{repo}/src") else []))
    env = dict(os.environ, PYTHONPATH=path, PYTHONHASHSEED=str(seed))
    cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors", f"--timeout={timeout_s}",
           "--junitxml", xml]
    cmd += ["-p", "randomly", f"--randomly-seed={seed}"] if randomize else ["-p", "no:randomly"]
    rc, out = gr.sh(cmd + [target_dir], cwd=repo, timeout=900, env=env)
    res = {}
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            tid = f'{tc.get("classname")}::{tc.get("name")}'; st = "pass"
            for ch in tc:
                if ch.tag in ("failure", "error", "skipped"):
                    st = {"failure": "fail", "error": "error", "skipped": "skip"}[ch.tag]
            res[tid] = st
    except Exception:
        pass
    finally:
        os.unlink(xml)
    return res, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--work", default="/tmp/matched_work"); ap.add_argument("--runs", type=int, default=150)
    ap.add_argument("--timeout", type=int, default=60); ap.add_argument("--out", required=True)
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
    res = {"project": a.project, "sha": r["Project_Hash"], "category": r["flakiness category"], "install_ok": ok, "suites": {}}
    # control: the known flaky human test, same number of runs, file order
    hc = collections.Counter(); hm = []
    for i in range(a.runs):
        out_res, _ = gr.run_once(py, repo, r["Test_funcname"], r["Test_filename"], 120)
        sts = [s for _, s, _ in out_res]
        st = "fail" if "fail" in sts else "error" if "error" in sts else "pass" if "pass" in sts else "skip"
        hc[st] += 1
        for _, s, m in out_res:
            if s in ("fail", "error") and m and m not in hm and len(hm) < 3: hm.append(m)
        if i == 0 and (not out_res or all(s == "error" for s in sts)): break
    res["human"] = {"runs": sum(hc.values()), "passed": hc["pass"], "failed": hc["fail"], "errors": hc["error"],
                    "reproduced": hc["pass"] > 0 and hc["fail"] + hc["error"] > 0, "messages": hm}
    for sdir in sorted(glob.glob(f"{ROOT}/matched/{a.project}/*/")):
        sid = os.path.basename(sdir.rstrip("/"))
        model, prompt, sample = (sid.split("_") + ["", "", ""])[:3]
        dst = f"{repo}/zz_{sid}"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
        for f in glob.glob(f"{sdir}/*.py"):
            if os.path.basename(f) not in ("__init__.py", "conftest.py"): shutil.copy(f, dst)
        hist = collections.defaultdict(collections.Counter); first_fail = []
        for i in range(a.runs):
            seed = 7000 + i
            out_res, tail = run_suite(py, repo, f"zz_{sid}", a.timeout, seed, randomize=(i % 2 == 1))
            if i == 0:
                first_fail = sorted(t for t, s in out_res.items() if s in ("fail", "error"))
            for t, s in out_res.items(): hist[t][s] += 1
        tests = {}
        for t, c in hist.items():
            p, f = c["pass"], c["fail"] + c["error"]
            kind = "flaky" if p and f else "always-fail" if f else "always-skip" if not p else "stable"
            tests[t] = {"pass": p, "fail": c["fail"], "error": c["error"], "skip": c["skip"], "kind": kind}
        res["suites"][sid] = {"model": model, "prompt": prompt, "sample": sample, "runs": a.runs, "n_tests": len(tests),
                              "first_run_failed": first_fail, "tests": tests,
                              "flaky": sorted(t for t, v in tests.items() if v["kind"] == "flaky"),
                              "always_fail": sorted(t for t, v in tests.items() if v["kind"] == "always-fail")}
        print(f"[{a.project}/{sid}] tests={len(tests)} flaky={len(res['suites'][sid]['flaky'])} "
              f"always_fail={len(res['suites'][sid]['always_fail'])}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
