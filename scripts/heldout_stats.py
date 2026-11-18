#!/usr/bin/env python3
"""The pre registered analysis of the held out sample (Progress Report 2, step 1).

Runs scripts/matched_stats.py exactly as it was at commit 7c05b6b, unmodified, on the held out suites alone. The
held out results (results/matched_heldout/*.json, suites <model>_heldout_<k>, <model>_heldoutseed_<k>,
<model>_heldoutiso_<k>) are copied into a scratch folder with the layout that version expects, with the prompt
labels heldout, heldoutseed, heldoutiso renamed to plain, plainseed, plainiso, and the frozen script is run there.
Its "all" block is the test we committed to: the per project exact sign test for each of the six model pairs,
significant when p < 0.05 / 6. Writes results/heldout_stats.json.

  .venv/bin/python scripts/heldout_stats.py
"""
import glob, json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = "7c05b6b"
RENAME = {"heldout": "plain", "heldoutseed": "plainseed", "heldoutiso": "plainiso"}


def main():
    tmp = tempfile.mkdtemp(prefix="heldout_stats_")
    os.makedirs(f"{tmp}/scripts"); os.makedirs(f"{tmp}/results/matched"); os.makedirs(f"{tmp}/results/gruber")
    for f in ("scripts/matched_stats.py", "scripts/analyze_matched.py"):
        src = subprocess.run(["git", "-C", ROOT, "show", f"{FROZEN}:{f}"], capture_output=True, text=True, check=True).stdout
        open(f"{tmp}/{f}", "w").write(src)
    shutil.copy(f"{ROOT}/results/gruber/matched_projects.txt", f"{tmp}/results/gruber/")
    n = 0
    for f in sorted(glob.glob(f"{ROOT}/results/matched_heldout/*.json")):
        j = json.load(open(f))
        suites = {}
        for sid, s in j["suites"].items():
            if s.get("prompt") in RENAME:
                s = dict(s, prompt=RENAME[s["prompt"]])
                suites[sid] = s; n += 1
        j["suites"] = suites
        json.dump(j, open(f"{tmp}/results/matched/{os.path.basename(f)}", "w"))
    p = subprocess.run([sys.executable, f"{tmp}/scripts/matched_stats.py"], capture_output=True, text=True)
    print(p.stdout[-4000:], p.stderr[-2000:])
    res = json.load(open(f"{tmp}/results/matched_stats.json"))
    alpha = 0.05 / 6
    res["preregistered"] = {"frozen_commit": FROZEN, "suites": n, "alpha": round(alpha, 4),
                            "significant_pairs": [k for k, v in res["all"]["pairs"].items() if v["p_project_sign"] < alpha]}
    json.dump(res, open(f"{ROOT}/results/heldout_stats.json", "w"), indent=1)
    print(json.dumps(res["preregistered"]))


if __name__ == "__main__":
    main()
