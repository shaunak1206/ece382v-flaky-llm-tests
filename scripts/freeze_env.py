#!/usr/bin/env python3
"""Record the package versions of a known flaky code study project's environment (Progress Report 3).

Builds the environment exactly as matched_run.py does (gruber_reproduce.install, the runner's extra packages, and
the missing module fixer) and writes `uv pip freeze` to results/envs/<project>.txt, so the environments the reported
runs used can be rebuilt. Recorded on the date in the file header; the runs used uv's newest compatible versions of the
same declared dependencies during November 15 to 19, 2026.

  python scripts/freeze_env.py <Project_Name> --out results/envs/<Project_Name>.txt
"""
import argparse, datetime, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruber_reproduce as gr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--work", default="/tmp/freeze"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = gr.row_for(a.project)
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
    rc, out = gr.sh(["uv", "pip", "freeze", "-p", py])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(f"# {a.project} @ {r['Project_Hash']}, Python 3.8, recorded {datetime.date.today()}\n" + out)


if __name__ == "__main__":
    main()
