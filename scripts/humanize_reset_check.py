#!/usr/bin/env python3
"""Direct test of the E3 explanation for the 10 flaky PL-G4 humanize tests.

Runs gen_plain_gpt41mini/humanize/ in shuffled order (pytest-randomly, seed 9000+i) N times twice: as generated,
and with a conftest.py that only adds an autouse fixture calling humanize.i18n.deactivate() before and after every
test (the reset that PL-S puts in every humanize file). If the explanation is right, the flaky tests flip without
the fixture and never with it. Writes results/humanize_reset_check.json.

  .venv/bin/python scripts/humanize_reset_check.py --runs 50
"""
import argparse, collections, json, os, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = f"{ROOT}/gen_plain_gpt41mini/humanize"
CONFTEST = '''import pytest
from humanize import i18n


@pytest.fixture(autouse=True)
def _reset_translation_state():
    i18n.deactivate()
    yield
    i18n.deactivate()
'''


def run(target, seed):
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "randomly",
                    f"--randomly-seed={seed}", "--junitxml", xml, target],
                   cwd=os.path.dirname(target), capture_output=True, text=True, timeout=600)
    out = {}
    for tc in ET.parse(xml).getroot().iter("testcase"):
        st = "pass"
        for ch in tc:
            if ch.tag in ("failure", "error"): st = "fail"
            elif ch.tag == "skipped": st = "skip"
        out[tc.get("name")] = st
    os.remove(xml)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=50); a = ap.parse_args()
    res = {}
    for variant in ("as_generated", "with_reset_fixture"):
        work = tempfile.mkdtemp(); target = f"{work}/suite"; shutil.copytree(SUITE, target)
        if variant == "with_reset_fixture":
            open(f"{target}/conftest.py", "w").write(CONFTEST)
        hist = collections.defaultdict(collections.Counter)
        for i in range(a.runs):
            for t, s in run(target, 9000 + i).items(): hist[t][s] += 1
        flaky = sorted(t for t, c in hist.items() if c["pass"] and c["fail"])
        res[variant] = {"runs": a.runs, "tests": len(hist), "flaky": flaky,
                        "always_fail": sorted(t for t, c in hist.items() if c["fail"] and not c["pass"]),
                        "failures_of_flaky": {t: hist[t]["fail"] for t in flaky}}
        print(variant, "tests", len(hist), "flaky", len(flaky), "always fail", len(res[variant]["always_fail"]), flush=True)
        shutil.rmtree(work)
    json.dump(res, open(f"{ROOT}/results/humanize_reset_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
