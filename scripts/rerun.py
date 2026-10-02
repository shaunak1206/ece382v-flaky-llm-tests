#!/usr/bin/env python3
"""Rerun a pytest target N times under perturbations and record per-test outcomes.

Modes:
  plain     - same order, fixed PYTHONHASHSEED=0
  order     - pytest-randomly with a different seed each run (random test order + reseeded RNGs)
  hashseed  - fixed order, but PYTHONHASHSEED varies each run (set/dict-hash iteration order)
  both      - random order AND varying PYTHONHASHSEED
  stress    - like both, plus CPU-burner processes running concurrently (timing pressure)
  tz        - like both, plus a different TZ (timezone) per run
"""
TZS = ["UTC", "America/Chicago", "Asia/Kolkata", "Pacific/Kiritimati", "Pacific/Pago_Pago",
       "Australia/Lord_Howe", "Europe/London", "Asia/Kathmandu", "America/St_Johns", "Etc/GMT-14"]
BURNER = "import time\nwhile True:\n    sum(i*i for i in range(10000))\n"
import argparse, json, os, subprocess, sys, tempfile, time, collections
import xml.etree.ElementTree as ET

def run_once(target, mode, seed, cwd, extra):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed if mode in ("hashseed", "both", "stress", "tz") else 0)
    if mode == "tz":
        env["TZ"] = TZS[seed % len(TZS)]
    burners = []
    lockf = None
    if mode == "stress":
        # only one stress run at a time machine wide, so stress means one burner per core
        import fcntl
        lockf = open("/Users/shaunkapoor/ece382v-flaky-llm-tests/results/stress.lock", "w")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        burners = [subprocess.Popen([sys.executable, "-c", BURNER]) for _ in range(os.cpu_count() or 4)]
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    if mode in ("order", "both", "stress", "tz"):
        cmd += ["-p", "randomly", f"--randomly-seed={seed}"]
    else:
        cmd += ["-p", "no:randomly"]
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    cmd += [f"--junitxml={xml}"] + extra + [target]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        proc = subprocess.CompletedProcess(cmd, -9, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "WHOLE RUN TIMED OUT after 900s")
    dt = time.time() - t0
    for b in burners: b.kill()
    if lockf: lockf.close()
    out = {}
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            tid = f'{tc.get("classname")}::{tc.get("name")}'
            status, msg = "pass", ""
            for child in tc:
                if child.tag in ("failure", "error", "skipped"):
                    status = {"failure": "fail", "error": "error", "skipped": "skip"}[child.tag]
                    msg = (child.get("message") or "")[:300]
            out[tid] = (status, msg)
    finally:
        os.unlink(xml)
    return out, dt, proc.returncode, (proc.stdout + proc.stderr)[-2000:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--mode", default="both", choices=["plain", "order", "hashseed", "both", "stress", "tz"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--deselect-file", default=None, help="file of pytest node ids to deselect (filtered-out generated tests)")
    ap.add_argument("extra", nargs="*", help="extra pytest args; put them after --")
    a = ap.parse_args()
    a.extra = [x for x in a.extra if x != "--"]
    if a.deselect_file and os.path.exists(a.deselect_file):
        for line in open(a.deselect_file):
            line=line.strip()
            if line: a.extra += ["--deselect", line]

    hist = collections.defaultdict(lambda: collections.Counter())
    msgs = collections.defaultdict(list)
    times = []; failing_runs = 0; ntests = None
    for i in range(a.runs):
        seed = 1000 + i
        res, dt, rc, tail = run_once(a.target, a.mode, seed, a.cwd, a.extra)
        times.append(dt)
        if rc not in (0, 1):  # 1 = tests failed, anything else = pytest crashed
            print(f"[{a.name}] run {i} pytest rc={rc}\n{tail}", file=sys.stderr)
        if not res:
            print(f"[{a.name}] run {i} produced no results\n{tail}", file=sys.stderr); continue
        ntests = len(res)
        if any(s in ("fail", "error") for s, _ in res.values()): failing_runs += 1
        for tid, (status, msg) in res.items():
            hist[tid][status] += 1
            if status in ("fail", "error") and len(msgs[tid]) < 3 and msg not in msgs[tid]:
                msgs[tid].append(msg)
        print(f"[{a.name}] run {i:3d} seed={seed} {dt:5.1f}s "
              f"fail/err={sum(1 for s,_ in res.values() if s in ('fail','error'))}", flush=True)

    tests = {}
    for tid, c in hist.items():
        passes = c["pass"]; bad = c["fail"] + c["error"]
        kind = "flaky" if passes and bad else ("always-fail" if bad else ("always-skip" if c["skip"] and not passes else "stable"))
        tests[tid] = {"counts": dict(c), "kind": kind, "messages": msgs.get(tid, [])}
    summary = {
        "name": a.name, "mode": a.mode, "runs": a.runs, "num_tests": ntests,
        "runs_with_failure": failing_runs, "avg_run_seconds": round(sum(times)/max(len(times),1), 2),
        "num_flaky": sum(1 for t in tests.values() if t["kind"] == "flaky"),
        "num_always_fail": sum(1 for t in tests.values() if t["kind"] == "always-fail"),
        "flaky_tests": sorted(t for t, v in tests.items() if v["kind"] == "flaky"),
        "tests": tests,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"[{a.name}] DONE mode={a.mode} tests={ntests} flaky={summary['num_flaky']} "
          f"always_fail={summary['num_always_fail']} runs_with_failure={failing_runs}/{a.runs}")
    for t in summary["flaky_tests"]:
        print(f"   FLAKY {t} {tests[t]['counts']} :: {tests[t]['messages'][:1]}")

if __name__ == "__main__":
    main()
