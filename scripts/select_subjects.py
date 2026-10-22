#!/usr/bin/env python3
"""Systematic subject selection funnel (answers the proposal feedback on how subjects are chosen).

Every step is a fixed, written down rule, the count after each step is logged, and the same script
can be rerun later to pull more subjects from the same pool.

  S0  candidate pool     the top N most downloaded PyPI packages over the last 30 days
                         (hugovk/top-pypi-packages snapshot, saved to results/subject_funnel/pool.json)
  S1  library            not a typing stub, pytest plugin, or Sphinx/flake8 plugin (name rules below)
  S2  source available   PyPI metadata links a GitHub repository
  S3  pure Python        the newest release ships a py3-none-any (or py2.py3-none-any) wheel and no
                         platform specific wheel, so no compiler toolchain is needed
  S4  small footprint    at most 3 required dependencies (extras not counted) and requires-python
                         allows 3.12, so the library can be installed and tested alone, offline
  S5  has a test suite   the GitHub repo (default branch, listed with a blobless git clone) contains
                         test_*.py or *_test.py files that pytest can collect (pytest also runs unittest)
  S6  runs fast, offline the suite passes at its pinned commit with sockets disabled (pytest-socket)
                         in under 60 seconds. This step clones and runs the suite, so it is applied to
                         a seeded random sample of S5 per stratum and to the current subjects.

Strata (assigned after S5 from the wheel source, purposeful oversampling, not a filter):
  ND  the library source imports a nondeterminism related module
      (time, datetime, random, threading, asyncio, concurrent, multiprocessing, tempfile, socket)
  D   it does not
Usage:
  python scripts/select_subjects.py --top 5000 --static        # S0 to S5 plus strata, cheap
  python scripts/select_subjects.py --run-sample 6 --seed 1    # S6 on 6 per stratum
"""
import argparse, io, json, os, random, re, shutil, subprocess, sys, tempfile, time, zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/results/subject_funnel"
POOL_URL = "https://hugovk.dev/top-pypi-packages/top-pypi-packages.min.json"
CURRENT = ["cachetools", "schedule", "tinydb", "humanize", "toolz", "tabulate", "tenacity", "filelock",
           "boltons", "more-itertools", "arrow", "pybreaker"]
ND_MODULES = ["time", "datetime", "random", "threading", "asyncio", "concurrent", "multiprocessing",
              "tempfile", "socket"]
NOT_LIBRARY = re.compile(r"^(types-|typing[-_]extensions$|pytest-|sphinx|flake8-|mypy-boto3|.*-stubs$|backports)", re.I)

def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "ece382v-subject-funnel"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def github_repo(info):
    urls = list((info.get("project_urls") or {}).values()) + [info.get("home_page") or ""]
    for u in urls:
        m = re.match(r"https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/|$|#)", u or "")
        if m and m.group(1).lower() not in ("sponsors", "orgs"):
            return f"{m.group(1)}/{m.group(2)}"
    return None

def required_deps(info):
    return [d for d in (info.get("requires_dist") or []) if "extra ==" not in d and "extra==" not in d]

def py312_ok(spec):
    if not spec: return True
    for part in spec.split(","):
        part = part.strip()
        m = re.match(r"(<=|<|==|!=|>=|>|~=)\s*3\.(\d+)", part)
        if not m: continue
        op, minor = m.group(1), int(m.group(2))
        if op == "<" and minor <= 12: return False
        if op == "<=" and minor < 12: return False
        if op == "==" and minor != 12 and "*" not in part: return False
    return True

def pypi_meta(name):
    try:
        j = get_json(f"https://pypi.org/pypi/{name}/json")
    except Exception as e:
        return {"name": name, "error": str(e)[:100]}
    info = j["info"]; files = j.get("urls", [])
    wheels = [f["filename"] for f in files if f["filename"].endswith(".whl")]
    pure = [w for w in wheels if re.search(r"-(py3|py2\.py3)-none-any\.whl$", w)]
    platform = [w for w in wheels if not w.endswith("-none-any.whl")]
    return {"name": name, "version": info.get("version"), "repo": github_repo(info),
            "pure_wheel": pure[0] if pure else None, "platform_wheels": len(platform),
            "wheel_url": next((f["url"] for f in files if pure and f["filename"] == pure[0]), None),
            "deps": required_deps(info), "requires_python": info.get("requires_python") or ""}

TREE_CACHE = f"{OUT}/trees.json"

def repo_tree(repo):
    """List the files of the repo default branch with a blobless partial clone (no GitHub API, no rate limit)."""
    d = tempfile.mkdtemp(prefix="tree_")
    try:
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
                            f"https://github.com/{repo}.git", d], capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        if r.returncode: return None
        paths = subprocess.run(["git", "-C", d, "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True).stdout.split()
        sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)
    tests = [x for x in paths if re.search(r"(^|/)(test_[^/]*|[^/]*_test)\.py$", x) and "/site-packages/" not in x]
    return {"sha": sha, "n_test_files": len(tests)}

def trees_for(repos):
    cache = json.load(open(TREE_CACHE)) if os.path.exists(TREE_CACHE) else {}
    todo = [r for r in repos if r not in cache or cache[r] is None]
    with ThreadPoolExecutor(16) as ex:
        for r, t in zip(todo, ex.map(repo_tree, todo)):
            cache[r] = t
    json.dump(cache, open(TREE_CACHE, "w"))
    return cache

def stratum(wheel_url):
    try:
        req = urllib.request.Request(wheel_url, headers={"User-Agent": "ece382v-subject-funnel"})
        data = urllib.request.urlopen(req, timeout=60).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        found = set()
        for n in z.namelist():
            if not n.endswith(".py") or "/tests/" in n or "/test_" in n: continue
            src = z.read(n).decode("utf-8", "replace")
            for m in ND_MODULES:
                if re.search(rf"^\s*(import {m}\b|from {m}\b)", src, re.M): found.add(m)
        return ("ND" if found else "D"), sorted(found)
    except Exception as e:
        return "?", [str(e)[:80]]

def static(top):
    os.makedirs(OUT, exist_ok=True)
    pool = get_json(POOL_URL)
    rows = pool["rows"]
    json.dump({"source": POOL_URL, "last_update": pool.get("last_update"), "rows": rows}, open(f"{OUT}/pool.json", "w"))
    rank = {r["project"].lower(): i + 1 for i, r in enumerate(rows)}
    names = [r["project"] for r in rows[:top]]
    # current subjects are always traced through the funnel, even if ranked below the cut
    trace = names + [c for c in CURRENT if c not in {n.lower() for n in names}]
    log = [("S0 top PyPI packages", len(names))]
    s1 = [n for n in trace if not NOT_LIBRARY.match(n)]
    with ThreadPoolExecutor(32) as ex:
        metas = {m["name"]: m for m in ex.map(pypi_meta, s1)}
    s2 = [n for n in s1 if metas[n].get("repo")]
    s3 = [n for n in s2 if metas[n].get("pure_wheel") and metas[n]["platform_wheels"] == 0]
    s4 = [n for n in s3 if len(metas[n]["deps"]) <= 3 and py312_ok(metas[n]["requires_python"])]
    cache = trees_for([metas[n]["repo"] for n in s4])
    trees = {n: cache.get(metas[n]["repo"]) for n in s4}
    s5 = [n for n in s4 if trees[n] and trees[n]["n_test_files"] > 0]
    with ThreadPoolExecutor(16) as ex:
        strata = dict(zip(s5, ex.map(lambda n: stratum(metas[n]["wheel_url"]), s5)))
    in_top = lambda lst: [n for n in lst if n in names]
    for label, lst in [("S1 is a library", s1), ("S2 has a GitHub repo", s2), ("S3 pure Python wheel", s3),
                       ("S4 <=3 deps and runs on 3.12", s4), ("S5 has a test suite", s5)]:
        log.append((label, len(in_top(lst))))
    nd = [n for n in in_top(s5) if strata[n][0] == "ND"]; d = [n for n in in_top(s5) if strata[n][0] == "D"]
    log.append(("  stratum ND (imports time/random/threads/...)", len(nd)))
    log.append(("  stratum D", len(d)))
    current = {}
    for c in CURRENT:
        key = next((n for n in trace if n.lower() == c), c)
        steps = {"S1": key in s1, "S2": key in s2, "S3": key in s3, "S4": key in s4, "S5": key in s5}
        first_fail = next((k for k, v in steps.items() if not v), None)
        current[c] = {"rank": rank.get(c), "passes_static": first_fail is None, "first_failed_step": first_fail,
                      "stratum": strata.get(key, ("-",))[0], "nd_imports": strata.get(key, ("-", []))[1],
                      "deps": metas.get(key, {}).get("deps")}
    res = {"top": top, "pool_last_update": pool.get("last_update"), "counts": log,
           "S5": [{"name": n, "rank": rank.get(n.lower()), "repo": metas[n]["repo"], "stratum": strata[n][0],
                   "nd_imports": strata[n][1], "n_test_files": trees[n]["n_test_files"], "deps": metas[n]["deps"]}
                  for n in in_top(s5)],
           "current_subjects": current}
    json.dump(res, open(f"{OUT}/static.json", "w"), indent=1)
    for k, v in log: print(f"{k:50s} {v:6d}")
    print("\ncurrent subjects:")
    for c, v in current.items():
        print(f"  {c:16s} rank={v['rank']} static={'pass' if v['passes_static'] else 'FAIL at ' + v['first_failed_step']} "
              f"stratum={v['stratum']} {v['nd_imports']}")

def run_suite(name, repo, workdir):
    """S6: shallow clone, install into a fresh uv venv, run pytest offline with a 60 s budget."""
    d = f"{workdir}/{name}"
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.time()
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/{repo}.git", d], capture_output=True, text=True)
    if r.returncode: return {"name": name, "ok": False, "why": "clone failed"}
    sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0", VIRTUAL_ENV=f"{d}/.venv")
    subprocess.run(["uv", "venv", "-q", "-p", "3.12", f"{d}/.venv"], capture_output=True)
    py = f"{d}/.venv/bin/python"
    inst = subprocess.run(["uv", "pip", "install", "-q", "-p", py, "-e", d, "pytest", "pytest-socket", "pytest-timeout",
                           "freezegun", "pytest-asyncio"], capture_output=True, text=True, env=env, cwd=d, timeout=600)
    if inst.returncode: return {"name": name, "sha": sha, "ok": False, "why": "install failed: " + inst.stderr[-200:]}
    testdir = "tests" if os.path.isdir(f"{d}/tests") else ("test" if os.path.isdir(f"{d}/test") else ".")
    try:
        p = subprocess.run([py, "-m", "pytest", "-q", "-x", "--disable-socket", "--allow-unix-socket", "-p", "no:cacheprovider",
                            "--timeout=60", testdir], capture_output=True, text=True, cwd=d, timeout=60, env=env)
        dt = time.time() - t0
        tail = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
        return {"name": name, "sha": sha, "ok": p.returncode == 0, "rc": p.returncode, "summary": tail[0][:200],
                "why": "" if p.returncode == 0 else ("failing or erroring tests" if p.returncode == 1 else f"pytest rc {p.returncode}")}
    except subprocess.TimeoutExpired:
        return {"name": name, "sha": sha, "ok": False, "why": "suite over 60 s"}

def run_sample(k, seed, include_current):
    st = json.load(open(f"{OUT}/static.json"))
    rng = random.Random(seed)
    current = {c.lower() for c in CURRENT}
    picks = []
    for s in ("ND", "D"):
        pool = [r for r in st["S5"] if r["stratum"] == s and r["name"].lower() not in current]
        picks += rng.sample(pool, min(k, len(pool)))
    work = tempfile.mkdtemp(prefix="funnel_")
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda r: dict(run_suite(r["name"], r["repo"], work), stratum=r["stratum"], rank=r["rank"]), picks))
    shutil.rmtree(work, ignore_errors=True)
    json.dump({"seed": seed, "per_stratum": k, "results": res}, open(f"{OUT}/s6_sample.json", "w"), indent=1)
    for r in res:
        print(f"  {r['stratum']} {r['name']:28s} rank={r['rank']} {'PASS' if r['ok'] else 'fail'} {r.get('why','')} {r.get('summary','')}")
    print(f"S6 pass: {sum(r['ok'] for r in res)} of {len(res)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5000)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--run-sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.static: static(a.top)
    if a.run_sample: run_sample(a.run_sample, a.seed, True)
