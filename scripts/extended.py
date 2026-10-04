#!/usr/bin/env python3
"""Extended, resumable rerun experiment (built to run unattended in chunks, e.g. on Colab).

For every suite it reruns only the tests the static scan flagged as risky, many times, cycling
through 10 time zones. Each (suite, time zone) chunk is one pytest process that repeats every
flagged test COUNT times in random order (pytest-repeat + pytest-randomly). Each finished chunk is
written to OUT as its own JSON file, so a disconnected run can simply be started again and it
skips what is already done.

usage: python scripts/extended.py --root <project dir> --subjects <subjects dir> --out <dir>
                                  [--count 100] [--workers N] [--conds human,llm_tests,...]
       python scripts/extended.py --summarize --out <dir>
"""
import argparse, collections, concurrent.futures as cf, glob, json, os, re, subprocess, sys, tempfile, time
import xml.etree.ElementTree as ET

TZS = ["UTC", "America/Chicago", "Asia/Kolkata", "Pacific/Kiritimati", "Pacific/Pago_Pago",
       "Australia/Lord_Howe", "Europe/London", "Asia/Kathmandu", "America/St_Johns", "Etc/GMT-14"]
HUMAN = {  # proj: (subject dir, extra pytest args, env)
    "cachetools": ("cachetools", [], {}), "schedule": ("schedule", [], {}), "tinydb": ("tinydb", ["-o", "addopts="], {}),
    "humanize": ("humanize", [], {}), "toolz": ("toolz", [], {}), "tabulate": ("python-tabulate", [], {}),
    "tenacity": ("tenacity", ["-o", "addopts="], {}), "filelock": ("filelock", ["-o", "addopts="], {"PYTHONPATH": "."}),
    "boltons": ("boltons", ["-o", "addopts="], {}), "more_itertools": ("more-itertools", ["-o", "addopts="], {}),
    "arrow": ("arrow", ["-o", "addopts="], {}), "pybreaker": ("pybreaker", ["-o", "addopts="], {}),
}


def flagged(root):
    """{(cond, proj): {file_path: set(test function names)}} from results/smells.json."""
    sm = json.load(open(f"{root}/results/smells.json"))
    out = collections.defaultdict(lambda: collections.defaultdict(set))
    for key, s in sm.items():
        cond, proj = key.split("/")
        for t in s["tests"]:
            if t["smells"]:
                rel = t["file"].split("/ece382v-flaky-llm-tests/", 1)[-1]  # make portable
                out[(cond, proj)][rel].add(t["test"])
    return out


def suite_paths(root, subjects, cond, proj, rel_file):
    if cond == "human":
        sd, extra, env = HUMAN[proj]
        cwd = f"{subjects}/{sd}"
        f = rel_file.split(f"subjects/{sd}/", 1)[-1]
        return cwd, f, extra, env
    return f"{root}/{cond}/{proj}", os.path.basename(rel_file), [], {}


def collect_nodeids(py, cwd, files, names, extra, env):
    p = subprocess.run([py, "-m", "pytest", "--collect-only", "-q", "-p", "no:randomly", "-p", "no:cacheprovider",
                        "--continue-on-collection-errors"] + extra + files, cwd=cwd, env=env, capture_output=True, text=True)
    ids = []
    for line in p.stdout.splitlines():
        if "::" not in line:
            continue
        fn = re.split(r"\[", line.split("::")[-1])[0]
        if fn in names:
            ids.append(line.strip())
    return ids


def run_chunk(job):
    py, cwd, ids, extra, env_extra, tz, seed, count, outp = job
    env = dict(os.environ, PYTHONHASHSEED=str(seed), TZ=tz, **env_extra)
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    cmd = [py, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-p", "randomly", f"--randomly-seed={seed}",
           f"--count={count}", "--timeout=15", "--continue-on-collection-errors", f"--junitxml={xml}"] + extra + ids
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=3 * 3600)
    except subprocess.TimeoutExpired:
        pass
    res = collections.defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0, "messages": []})
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            name = re.sub(r"-?\d+-\d+\]$", "]", tc.get("name") or "").replace("[]", "")  # strip pytest-repeat suffix
            tid = f'{tc.get("classname")}::{name}'
            st = "pass"
            for ch in tc:
                if ch.tag in ("failure", "error"): st = "fail"
                elif ch.tag == "skipped": st = "skip"
            r = res[tid]; r[st] += 1
            if st == "fail" and len(r["messages"]) < 2:
                m = (tc.find("failure") if tc.find("failure") is not None else tc.find("error")).get("message", "")[:300]
                if m not in r["messages"]: r["messages"].append(m)
    except Exception as e:  # junit missing, e.g. the whole chunk hung
        res["__chunk_error__"]["messages"].append(repr(e))
    os.unlink(xml) if os.path.exists(xml) else None
    tmp = outp + ".tmp"
    json.dump({"tz": tz, "seed": seed, "count": count, "seconds": round(time.time() - t0, 1), "tests": res}, open(tmp, "w"))
    os.replace(tmp, outp)  # atomic, so a disconnect never leaves a half written chunk
    return outp, round(time.time() - t0, 1)


def summarize(out):
    agg = collections.defaultdict(lambda: {"pass": 0, "fail": 0, "tz_fail": collections.Counter(), "messages": []})
    chunks = sorted(glob.glob(f"{out}/*.json"))
    for f in chunks:
        if os.path.basename(f) == "summary.json":
            continue
        d = json.load(open(f)); suite = os.path.basename(f).split("__tz")[0]
        for tid, v in d["tests"].items():
            a = agg[(suite, tid)]; a["pass"] += v.get("pass", 0); a["fail"] += v.get("fail", 0)
            if v.get("fail"): a["tz_fail"][d["tz"]] += v["fail"]
            for m in v.get("messages", []):
                if m not in a["messages"] and len(a["messages"]) < 2: a["messages"].append(m)
    flaky = {f"{s}::{t}": {k: (dict(v) if k == "tz_fail" else v) for k, v in a.items()}
             for (s, t), a in agg.items() if a["pass"] and a["fail"]}
    always = sum(1 for a in agg.values() if a["fail"] and not a["pass"])
    by_suite = collections.Counter(k.split("::")[0].rsplit("__", 0)[0].split("__")[0] for k in flaky)
    summ = {"chunks": len(chunks), "tests": len(agg), "executions": sum(a["pass"] + a["fail"] for a in agg.values()),
            "flaky": len(flaky), "always_fail": always, "flaky_by_condition": dict(by_suite), "flaky_tests": flaky}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "flaky_tests"}, indent=1))
    for k, v in flaky.items():
        print(f"FLAKY {k} pass={v['pass']} fail={v['fail']} tz={v['tz_fail']} :: {(v['messages'] or [''])[0][:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--conds", default="human,llm_tests,gen_oneshot_sonnet,gen_oneshot_haiku")
    ap.add_argument("--only", default=None, help="comma list of projects to limit to (for testing)")
    ap.add_argument("--tzs", type=int, default=len(TZS), help="use only the first N time zones (for testing)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        return summarize(a.out)
    subjects = a.subjects or f"{a.root}/subjects"
    py = sys.executable
    jobs = []
    for (cond, proj), files in sorted(flagged(a.root).items()):
        if cond not in a.conds.split(",") or (a.only and proj not in a.only.split(",")):
            continue
        names = set().union(*files.values())
        cwd, _, extra, env = suite_paths(a.root, subjects, cond, proj, next(iter(files)))
        fl = [suite_paths(a.root, subjects, cond, proj, f)[1] for f in files]
        ids = collect_nodeids(py, cwd, fl, names, extra, dict(os.environ, **env))
        if not ids:
            print(f"[skip] {cond}/{proj}: no flagged node ids collected"); continue
        for i, tz in enumerate(TZS[: a.tzs]):
            outp = f"{a.out}/{cond}__{proj}__tz{i}.json"
            if os.path.exists(outp):
                continue
            jobs.append((py, cwd, ids, extra, env, tz, 5000 + i, a.count, outp))
        print(f"[plan] {cond}/{proj}: {len(ids)} flagged tests")
    print(f"[plan] {len(jobs)} chunks to run with {a.workers} workers, count={a.count}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for done, (outp, secs) in enumerate(ex.map(run_chunk, jobs), 1):
            print(f"[{time.strftime('%H:%M')}] {done}/{len(jobs)} {os.path.basename(outp)} {secs}s", flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
