#!/usr/bin/env python3
"""Static habits of the plain prompt matched suites (E3): how each model handles nondeterminism.

For every matched/<project>/<model>_plain_<k>/test_gen.py, records whether the suite
  seeds:      calls random.seed, numpy.random.seed, default_rng(<arg>), RandomState(<arg>) or torch.manual_seed
  monkeypatch uses the monkeypatch fixture or unittest.mock.patch
  reloads:    calls importlib.reload or reads or writes sys.modules
and counts suites per model, for all 41 projects and for the first 17. Writes results/matched_habits.json.

  python scripts/matched_habits.py
"""
import collections, glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HABITS = {
    "seeds": re.compile(r"random\.seed\(|default_rng\(\s*\w|RandomState\(\s*\w|manual_seed\("),
    "monkeypatch": re.compile(r"\bmonkeypatch\b|mock\.patch|@patch\("),
    "reloads": re.compile(r"importlib\.reload|\breload\(|sys\.modules"),
}


def main():
    first = set(open(f"{ROOT}/results/gruber/matched_projects.txt").read().split()[:17])
    out = {"all": collections.defaultdict(collections.Counter), "first_17": collections.defaultdict(collections.Counter)}
    for f in sorted(glob.glob(f"{ROOT}/matched/*/*_plain_*/test_gen.py")):
        project, sid = f.split("/")[-3:-1]
        model = sid.split("_")[0]; src = open(f).read()
        for scope in ("all", "first_17"):
            if scope == "first_17" and project not in first: continue
            c = out[scope][model]; c["suites"] += 1
            for h, rx in HABITS.items():
                c[h] += bool(rx.search(src))
    json.dump(out, open(f"{ROOT}/results/matched_habits.json", "w"), indent=1)
    for scope, by in out.items():
        print(scope, {m: dict(c) for m, c in by.items()})


if __name__ == "__main__":
    main()
