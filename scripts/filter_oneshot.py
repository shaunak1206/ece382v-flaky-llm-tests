#!/usr/bin/env python3
"""Single-run generate-and-filter for one-shot suites.

Runs a generated suite once in plain mode (fixed order, PYTHONHASHSEED=0) inside an isolated
work dir and records which tests a TestPilot style pipeline would keep (passed) or discard
(failed, errored, or in a file that does not even import). Writes results/filter/<cond>/<proj>.json
"""
import json, os, subprocess, sys, tempfile, xml.etree.ElementTree as ET
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(cond, proj):
    suite = f"{ROOT}/{cond}/{proj}"
    work = f"{ROOT}/results/work/{cond}/{proj}/filter"; os.makedirs(work, exist_ok=True)
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    env = dict(os.environ, PYTHONHASHSEED="0")
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-p", "no:randomly",
           "--continue-on-collection-errors", "--timeout=60", f"--junitxml={xml}", suite]
    p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
    passed, rejected, coll_err_files = [], [], []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        tid = f'{tc.get("classname")}::{tc.get("name")}'
        bad = [c for c in tc if c.tag in ("failure", "error")]
        skip = [c for c in tc if c.tag == "skipped"]
        if not tc.get("name") or (bad and tc.get("name") in ("", None)):
            continue
        if bad and tc.get("classname", "").count(".") == 0 and tc.get("name").endswith(".py"):
            coll_err_files.append(tc.get("name")); continue
        (rejected if bad else passed).append(tid)
    os.unlink(xml)
    # collection errors appear as a testcase whose classname is the file path without ::
    for line in (p.stdout + p.stderr).splitlines():
        if line.startswith("ERROR ") and "::" not in line:
            f = line.split()[1]
            if f not in coll_err_files: coll_err_files.append(f)
    out = {"cond": cond, "proj": proj, "collected": len(passed) + len(rejected),
           "accepted": len(passed), "rejected": len(rejected), "collection_error_files": coll_err_files,
           "accepted_ids": passed, "rejected_ids": rejected, "tail": (p.stdout + p.stderr)[-1500:]}
    os.makedirs(f"{ROOT}/results/filter/{cond}", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/filter/{cond}/{proj}.json", "w"), indent=1)
    print(f"[filter {cond}/{proj}] collected={out['collected']} accepted={out['accepted']} "
          f"rejected={out['rejected']} collection_error_files={len(coll_err_files)}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
