#!/usr/bin/env python3
"""iFixFlakies style cleaner search for the order dependent generated tests (Progress Report 3).

For one project, for every victim the polluter search (od_diagnose.py) found, in results/od_diagnose_heldout*,
od_diagnose_od*:
  confirm:  run (polluter, victim) 10 times with PYTHONHASHSEED=0 in that order; the polluter is confirmed if the
            victim fails at least 8 times, and the victim passes alone 3 of 3 times.
  cleaner:  iFixFlakies (Shi et al., FSE 2019) fixes a victim by running the statements of a cleaner, a test of the
            same suite that resets the state the polluter changed. For each other test c of the suite, run
            (polluter, c, victim); c is a cleaner if the victim then passes 3 of 3 times.
A victim with a cleaner can be fixed by an iFixFlakies patch. We compare that with the M2 fixture, which needs no
cleaner. Writes results/cleaners/<project>.json.

  python scripts/cleaner_search.py <Project_Name> --out results/cleaners/<Project_Name>.json
"""
import argparse, glob, json, os, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruber_reproduce as gr
from od_diagnose import node_id, run

ROOT = gr.ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--work", default="/tmp/cleaners"); ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=int, default=330)
    a = ap.parse_args()
    victims = []
    for d in ("od_diagnose_heldout", "od_diagnose_heldout_gpt", "od_diagnose_od", "od_diagnose_od_gpt"):
        f = f"{ROOT}/results/{d}/{a.project}.json"
        if not os.path.exists(f): continue
        for t in json.load(open(f))["tests"]:
            if t["verdict"] == "od" and t["polluters"]:
                victims.append((d, t["suite"], t["test"], t["polluters"]))
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
    out = {"project": a.project, "install_ok": ok, "victims": []}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    results_by_suite = {}
    for src in glob.glob(f"{ROOT}/results/matched_heldout/{a.project}.json") + glob.glob(f"{ROOT}/results/matched_od/{a.project}.json"):
        results_by_suite.update(json.load(open(src))["suites"])
    for sid in sorted({v[1] for v in victims}):
        dst = f"{repo}/zz_{sid}"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
        for f in glob.glob(f"{ROOT}/matched/{a.project}/{sid}/*.py"):
            if os.path.basename(f) not in ("__init__.py", "conftest.py"): shutil.copy(f, dst)
    for d_, sid, test, pols in victims:
        tests = results_by_suite.get(sid, {}).get("tests", {})
        ids = {t.split("::")[-1]: t for t in tests}
        vid = ids.get(test)
        if not vid: continue
        pol = next((p for p in pols if p in ids), None)
        rec = {"source": d_, "suite": sid, "test": test, "polluter": pol}
        if not pol: out["victims"].append(rec); continue
        v, p = node_id(vid), node_id(ids[pol])
        rec["alone_pass"] = sum(run(py, repo, [v], 0) == "PASSED" for _ in range(3))
        rec["after_polluter_fail"] = sum(run(py, repo, [p, v], 0) != "PASSED" for _ in range(10))
        rec["confirmed"] = rec["alone_pass"] == 3 and rec["after_polluter_fail"] >= 8
        rec["cleaners"] = []
        if rec["confirmed"]:
            for name, tid in ids.items():
                if name in (test, pol) or tid.startswith("::"): continue
                c = node_id(tid)
                if run(py, repo, [p, c, v], 0) == "PASSED" and all(run(py, repo, [p, c, v], 0) == "PASSED" for _ in range(2)):
                    rec["cleaners"].append(name)
                if time.time() - t0 > a.max_minutes * 60: break
        out["victims"].append(rec)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"[{a.project}/{sid}] {test} confirmed={rec.get('confirmed')} cleaners={len(rec.get('cleaners', []))}", flush=True)
        if time.time() - t0 > a.max_minutes * 60:
            out["stopped_early"] = True; break
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
