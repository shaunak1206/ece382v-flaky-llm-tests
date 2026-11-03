#!/usr/bin/env python3
"""Reproduce one known flaky test from the Gruber et al. (ICST 2021) dataset.

Input is one row of results/gruber/candidates.csv (picked by project name). The project is checked out at
the commit Gruber et al. used, installed into a fresh Python 3.8 virtual environment (the version FlaPy
used), and the known flaky test is run many times. A test counts as reproduced when it both passes and
fails on our runner.

  python scripts/gruber_reproduce.py <Project_Name> --work /tmp/gr --runs 150 --out results/gruber/repro/<name>.json

Output JSON fields
  project, url, sha, test (pytest -k expression used), category (Gruber manual label, or 'unlabeled' for candidates_ext.csv),
  gruber_fail_rate (failures / runs in the original study, same order mode),
  install_ok, install_log_tail, collected (number of test ids the -k expression selected),
  runs, passed, failed, errors (counts over the repeated runs), reproduced (passed > 0 and failed+errors > 0),
  messages (up to 3 distinct failure messages), seconds
The project checkout stays in <work>/<name>/repo so the generation step can read the code under test.
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, time, tempfile
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ_FILES = ["requirements.txt", "requirements-dev.txt", "requirements_dev.txt", "dev-requirements.txt",
             "test-requirements.txt", "requirements-test.txt", "requirements_test.txt", "requirements/test.txt",
             "requirements/dev.txt", "tests/requirements.txt"]


def row_for(name):
    # candidates.csv: the 58 hand labeled tests; candidates_ext.csv: one unlabeled flaky test per remaining project
    for f in ("candidates.csv", "candidates_ext.csv"):
        for r in csv.DictReader(open(f"{ROOT}/results/gruber/{f}")):
            if r["Project_Name"] == name:
                return r
    raise SystemExit(f"unknown project {name}")


def sh(cmd, cwd=None, timeout=900, env=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        return -9, f"TIMEOUT after {timeout}s"


def install(repo, venv, log):
    py = f"{venv}/bin/python"
    rc, out = sh(["uv", "venv", "-q", "-p", "3.8", venv]); log.append(out[-500:])
    if rc: return False, py
    base = ["uv", "pip", "install", "-q", "-p", py]
    rc, out = sh(base + ["pytest<8", "pytest-timeout", "setuptools<70", "wheel"]); log.append(out[-500:])
    for f in REQ_FILES:
        if os.path.exists(f"{repo}/{f}"):
            rc, out = sh(base + ["-r", f], cwd=repo); log.append(f"[{f}] rc={rc} " + out[-800:])
    if any(os.path.exists(f"{repo}/{f}") for f in ("setup.py", "pyproject.toml", "setup.cfg")):
        ok = False
        for extra in ("[test,tests,testing,dev]", ""):
            rc, out = sh(base + ["-e", f".{extra}"], cwd=repo); log.append(f"[-e .{extra}] rc={rc} " + out[-800:])
            if rc == 0: ok = True; break
        if not ok:
            rc, out = sh(base + ["."], cwd=repo); log.append(f"[.] rc={rc} " + out[-800:])
    return True, py


PIP_NAME = {"sklearn": "scikit-learn", "cv2": "opencv-python-headless", "yaml": "pyyaml", "PIL": "pillow",
            "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "skimage": "scikit-image", "Crypto": "pycryptodome",
            "dotenv": "python-dotenv", "jwt": "pyjwt", "magic": "python-magic", "attr": "attrs", "google.protobuf": "protobuf"}


def fix_missing_modules(py, repo, testfile, log, tries=8):
    """Collect the test file, and pip install any module the collection says is missing (test-only dependencies)."""
    env = dict(os.environ, PYTHONPATH=repo)
    for _ in range(tries):
        rc, out = sh([py, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", testfile], cwd=repo, timeout=300, env=env)
        m = re.search(r"No module named '([\w\.]+)'", out)
        if not m:
            return out
        mod = m.group(1); top = mod.split(".")[0]
        pkg = PIP_NAME.get(mod, PIP_NAME.get(top, top))
        rc2, out2 = sh(["uv", "pip", "install", "-q", "-p", py, pkg], cwd=repo); log.append(f"[auto {pkg}] rc={rc2} " + out2[-300:])
        if rc2:
            return out
    return out


def run_once(py, repo, kexpr, testfile, timeout_s):
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    env = dict(os.environ, PYTHONPATH=repo)
    rc, out = sh([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", f"--timeout={timeout_s}",
                  "--junitxml", xml, "-k", kexpr, testfile], cwd=repo, timeout=timeout_s * 4 + 60, env=env)
    res = []
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            st, msg = "pass", ""
            for ch in tc:
                if ch.tag in ("failure", "error", "skipped"):
                    st = {"failure": "fail", "error": "error", "skipped": "skip"}[ch.tag]; msg = (ch.get("message") or "")[:200]
            res.append((tc.get("name"), st, msg))
    except Exception:
        res = [("<no junit>", "error", out[-200:])]
    finally:
        os.unlink(xml)
    return res, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project"); ap.add_argument("--work", default="/tmp/gruber_work")
    ap.add_argument("--runs", type=int, default=150); ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=int, default=180, help="stop rerunning after this much wall time")
    a = ap.parse_args()
    r = row_for(a.project); t0 = time.time()
    d = f"{a.work}/{a.project}"; repo = f"{d}/repo"; venv = f"{d}/venv"
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    sh(["git", "init", "-q", repo]); sh(["git", "-C", repo, "remote", "add", "origin", r["Project_URL"]])
    rc, out = sh(["git", "-C", repo, "fetch", "-q", "--depth", "1", "origin", r["Project_Hash"]], env=env)
    res = {"project": a.project, "url": r["Project_URL"], "sha": r["Project_Hash"], "category": r["flakiness category"],
           "gruber_fail_rate": float(r["fail_rate"] or 0), "test_file": r["Test_filename"], "test_func": r["Test_funcname"]}
    if rc:
        res.update(install_ok=False, install_log_tail="clone failed: " + out[-300:], reproduced=False)
        json.dump(res, open(a.out, "w"), indent=1); return
    sh(["git", "-C", repo, "checkout", "-q", "FETCH_HEAD"])
    log = []
    ok, py = install(repo, venv, log)
    kexpr = r["Test_funcname"]
    coll = fix_missing_modules(py, repo, r["Test_filename"], log)
    res.update(install_ok=ok, install_log_tail="\n".join(log)[-1500:], test=kexpr, collect_tail=coll[-600:])
    if not ok:
        res["reproduced"] = False; json.dump(res, open(a.out, "w"), indent=1); return
    counts = {"pass": 0, "fail": 0, "error": 0, "skip": 0}; msgs = []; collected = 0; runs = 0
    for i in range(a.runs):
        out_res, tail = run_once(py, repo, kexpr, r["Test_filename"], a.timeout)
        runs += 1
        if i == 0:
            collected = len(out_res); res["first_run_tail"] = tail[-800:]
            if collected == 0 or all(s == "error" for _, s, _ in out_res):
                counts["error"] += 1; msgs = [m for _, _, m in out_res][:3]
                break
        # a run fails if any selected test case fails
        sts = [s for _, s, _ in out_res]
        st = "fail" if "fail" in sts else "error" if "error" in sts else "pass" if "pass" in sts else "skip"
        counts[st] += 1
        for _, s, m in out_res:
            if s in ("fail", "error") and m and m not in msgs and len(msgs) < 3: msgs.append(m)
        if time.time() - t0 > a.max_minutes * 60: break
    res.update(collected=collected, runs=runs, passed=counts["pass"], failed=counts["fail"], errors=counts["error"],
               skipped=counts["skip"], messages=msgs, seconds=round(time.time() - t0, 1),
               reproduced=counts["pass"] > 0 and (counts["fail"] + counts["error"]) > 0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("project", "install_ok", "collected", "runs", "passed", "failed", "errors", "reproduced")}))


if __name__ == "__main__":
    main()
