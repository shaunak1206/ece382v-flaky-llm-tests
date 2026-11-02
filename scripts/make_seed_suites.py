#!/usr/bin/env python3
"""Make the M1 mitigation suites: every plain matched suite, unchanged, plus mitigations/seed_conftest.py.

  python scripts/make_seed_suites.py   -> matched/<project>/<model>_plainseed_<k>/{test_gen.py, conftest.py}
"""
import glob, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n = 0
for sdir in sorted(glob.glob(f"{ROOT}/matched/*/*_plain_*/")):
    model, _, sample = os.path.basename(sdir.rstrip("/")).split("_")
    dst = f"{os.path.dirname(sdir.rstrip('/'))}/{model}_plainseed_{sample}"
    os.makedirs(dst, exist_ok=True)
    shutil.copy(f"{sdir}/test_gen.py", f"{dst}/test_gen.py")
    shutil.copy(f"{ROOT}/mitigations/seed_conftest.py", f"{dst}/conftest.py")
    n += 1
print(n, "suites")
