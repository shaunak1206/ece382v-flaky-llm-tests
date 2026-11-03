#!/usr/bin/env python3
"""Make the mitigation suites: every plain matched suite, unchanged, plus a conftest.py added after generation.

  python scripts/make_seed_suites.py   -> matched/<project>/<model>_plainseed_<k>/  (M1, mitigations/seed_conftest.py)
                                          matched/<project>/<model>_plainiso_<k>/   (M2, mitigations/isolate_conftest.py)
"""
import glob, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n = 0
for sdir in sorted(glob.glob(f"{ROOT}/matched/*/*_plain_*/")):
    model, _, sample = os.path.basename(sdir.rstrip("/")).split("_")
    for prompt, conftest in (("plainseed", "seed_conftest.py"), ("plainiso", "isolate_conftest.py")):
        dst = f"{os.path.dirname(sdir.rstrip('/'))}/{model}_{prompt}_{sample}"
        os.makedirs(dst, exist_ok=True)
        shutil.copy(f"{sdir}/test_gen.py", f"{dst}/test_gen.py")
        shutil.copy(f"{ROOT}/mitigations/{conftest}", f"{dst}/conftest.py")
        n += 1
print(n, "suites")
