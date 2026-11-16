#!/usr/bin/env python3
"""Make the mitigation suites: every plain matched suite, unchanged, plus a conftest.py added after generation.

  python scripts/make_seed_suites.py   -> matched/<project>/<model>_plainseed_<k>/  (M1, mitigations/seed_conftest.py)
                                          matched/<project>/<model>_plainiso_<k>/   (M2, mitigations/isolate_conftest.py)
  python scripts/make_seed_suites.py heldout
                                       -> <model>_heldoutseed_<k>/ and <model>_heldoutiso_<k>/ from the held out suites
"""
import glob, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n = 0
src = sys.argv[1] if len(sys.argv) > 1 else "plain"
for sdir in sorted(glob.glob(f"{ROOT}/matched/*/*_{src}_*/")):
    model, _, sample = os.path.basename(sdir.rstrip("/")).split("_")
    for prompt, conftest in ((src + "seed", "seed_conftest.py"), (src + "iso", "isolate_conftest.py")):
        dst = f"{os.path.dirname(sdir.rstrip('/'))}/{model}_{prompt}_{sample}"
        os.makedirs(dst, exist_ok=True)
        shutil.copy(f"{sdir}/test_gen.py", f"{dst}/test_gen.py")
        shutil.copy(f"{ROOT}/mitigations/{conftest}", f"{dst}/conftest.py")
        n += 1
print(n, "suites")
