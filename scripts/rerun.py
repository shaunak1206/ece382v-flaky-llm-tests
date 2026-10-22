#!/usr/bin/env python3
"""Rerun a pytest target N times under perturbations and record per-test outcomes.

Modes:
  plain     - same order, fixed PYTHONHASHSEED=0
  order     - pytest-randomly with a different seed each run (random test order + reseeded RNGs)
  hashseed  - fixed order, but PYTHONHASHSEED varies each run (set/dict-hash iteration order)
  both      - random order AND varying PYTHONHASHSEED
  stress    - like both, plus CPU-burner processes running concurrently (timing pressure)
  tz        - like both, plus a different TZ (timezone) per run
  stressng  - like both, plus stress-ng CPU and memory noise. Following Shaker (Silva et al., ICSME 2020)
              the noise generator has four options, --cpu n, --cpu-load p, --vm n, --vm-bytes p. We use
              Shaker's sampling step, NOISE_CONFIGS drawn uniformly from those domains with a fixed seed,
              and run i uses config i mod len(NOISE_CONFIGS). The config of every run is recorded so the
              Shaker probability matrix (config x flaky test) can be built later.
"""
TZS = ["UTC", "America/Chicago", "Asia/Kolkata", "Pacific/Kiritimati", "Pacific/Pago_Pago",
       "Australia/Lord_Howe", "Europe/London", "Asia/Kathmandu", "America/St_Johns", "Etc/GMT-14"]
BURNER = "import time\nwhile True:\n    sum(i*i for i in range(10000))\n"
import argparse, json, os, random, shutil, subprocess, sys, tempfile, time, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCPU = os.cpu_count() or 4

def noise_configs(k=10, seed=2026):
    """Shaker style uniform sample over (cpu n, cpu-load %, vm n, vm-bytes %), scaled to this machine."""
    r = random.Random(seed)
    return [{"cpu": r.randint(1, 2 * NCPU), "cpu_load": r.randrange(10, 101, 10),
             "vm": r.randint(0, max(1, NCPU // 2)), "vm_bytes": r.randrange(10, 51, 10)} for _ in range(k)]
NOISE_CONFIGS = noise_configs()

def stressng_cmd(c):
    cmd = [shutil.which("stress-ng") or "stress-ng", "--quiet", "--timeout", "1000s",
           "--cpu", str(c["cpu"]), "--cpu-load", str(c["cpu_load"])]
    if c["vm"]:
        cmd += ["--vm", str(c["vm"]), "--vm-bytes", f'{c["vm_bytes"]}%']
    return cmd
import xml.etree.ElementTree as ET

def run_once(target, mode, seed, cwd, extra, run_index=0):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed if mode in ("hashseed", "both", "stress", "tz", "stressng") else 0)
    if mode == "tz":
        env["TZ"] = TZS[seed % len(TZS)]
    burners = []
    lockf = None
    noise = None
    if mode in ("stress", "stressng"):
        # only one stress run at a time machine wide
        import fcntl
        os.makedirs(f"{ROOT}/results", exist_ok=True)
        lockf = open(f"{ROOT}/results/stress.lock", "w")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if mode == "stress":  # one python busy loop per core
            burners = [subprocess.Popen([sys.executable, "-c", BURNER]) for _ in range(NCPU)]
        else:
            noise = NOISE_CONFIGS[run_index % len(NOISE_CONFIGS)]
            burners = [subprocess.Popen(stressng_cmd(noise), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        start_new_session=True)]
            time.sleep(1.0)  # let the stressors spin up
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    if mode in ("order", "both", "stress", "tz", "stressng"):
        cmd += ["-p", "randomly", f"--randomly-seed={seed}"]
    else:
        cmd += ["-p", "no:randomly"]
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    cmd += [f"--junitxml={xml}"] + extra + (target if isinstance(target, list) else [target])
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        proc = subprocess.CompletedProcess(cmd, -9, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "WHOLE RUN TIMED OUT after 900s")
    dt = time.time() - t0
    for b in burners:
        if noise is not None:
            import signal
            try: os.killpg(b.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        else:
            b.kill()
        b.wait()
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
    return out, dt, proc.returncode, (proc.stdout + proc.stderr)[-2000:], noise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", required=True, help="pytest target; several targets can be joined with ',' (mixed suites)")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--mode", default="both", choices=["plain", "order", "hashseed", "both", "stress", "tz", "stressng"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--deselect-file", default=None, help="file of pytest node ids to deselect (filtered-out generated tests)")
    ap.add_argument("extra", nargs="*", help="extra pytest args; put them after --")
    a = ap.parse_args()
    a.extra = [x for x in a.extra if x != "--"]
    if a.deselect_file and os.path.exists(a.deselect_file):
        for line in open(a.deselect_file):
            line=line.strip()
            if line: a.extra += ["--deselect", line]

    target = a.target.split(",") if "," in a.target else a.target
    hist = collections.defaultdict(lambda: collections.Counter())
    fail_runs = collections.defaultdict(list)  # test id -> indices of runs where it failed
    run_noise = []
    msgs = collections.defaultdict(list)
    times = []; failing_runs = 0; ntests = None
    for i in range(a.runs):
        seed = 1000 + i
        res, dt, rc, tail, noise = run_once(target, a.mode, seed, a.cwd, a.extra, i)
        times.append(dt); run_noise.append(noise)
        if rc not in (0, 1):  # 1 = tests failed, anything else = pytest crashed
            print(f"[{a.name}] run {i} pytest rc={rc}\n{tail}", file=sys.stderr)
        if not res:
            print(f"[{a.name}] run {i} produced no results\n{tail}", file=sys.stderr); continue
        ntests = len(res)
        if any(s in ("fail", "error") for s, _ in res.values()): failing_runs += 1
        for tid, (status, msg) in res.items():
            hist[tid][status] += 1
            if status in ("fail", "error"): fail_runs[tid].append(i)
            if status in ("fail", "error") and len(msgs[tid]) < 3 and msg not in msgs[tid]:
                msgs[tid].append(msg)
        print(f"[{a.name}] run {i:3d} seed={seed} {dt:5.1f}s "
              f"fail/err={sum(1 for s,_ in res.values() if s in ('fail','error'))}", flush=True)

    tests = {}
    for tid, c in hist.items():
        passes = c["pass"]; bad = c["fail"] + c["error"]
        kind = "flaky" if passes and bad else ("always-fail" if bad else ("always-skip" if c["skip"] and not passes else "stable"))
        tests[tid] = {"counts": dict(c), "kind": kind, "messages": msgs.get(tid, [])}
        if kind == "flaky": tests[tid]["fail_runs"] = fail_runs[tid]
    summary = {
        "name": a.name, "mode": a.mode, "runs": a.runs, "num_tests": ntests,
        "runs_with_failure": failing_runs, "avg_run_seconds": round(sum(times)/max(len(times),1), 2),
        "num_flaky": sum(1 for t in tests.values() if t["kind"] == "flaky"),
        "num_always_fail": sum(1 for t in tests.values() if t["kind"] == "always-fail"),
        "flaky_tests": sorted(t for t, v in tests.items() if v["kind"] == "flaky"),
        "tests": tests,
    }
    if a.mode == "stressng": summary["noise_per_run"] = run_noise
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"[{a.name}] DONE mode={a.mode} tests={ntests} flaky={summary['num_flaky']} "
          f"always_fail={summary['num_always_fail']} runs_with_failure={failing_runs}/{a.runs}")
    for t in summary["flaky_tests"]:
        print(f"   FLAKY {t} {tests[t]['counts']} :: {tests[t]['messages'][:1]}")

if __name__ == "__main__":
    main()
