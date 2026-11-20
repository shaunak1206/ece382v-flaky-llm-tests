#!/usr/bin/env python3
"""Summarize results/libfix (scripts/library_fixtures.sh): M1 and M2 on the library study suites.

Per condition and fixture: suites, tests, flaky tests, and, against the plain runs of the same suite, the stable tests
a fixture breaks (pass all plain runs, fail all fixture runs) and the flaky tests it removes or adds.
Writes results/library_fixtures_summary.json.

  python scripts/library_fixtures_summary.py
"""
import collections, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(f):
    j = json.load(open(f)); out = {}
    for t, v in j["tests"].items():
        out[t.split(".", 1)[1] if "." in t else t] = v["kind"]
    return out


def main():
    tot = collections.defaultdict(collections.Counter); flaky_list = []
    for f in sorted(glob.glob(f"{ROOT}/results/libfix/*__plain.json")):
        base = os.path.basename(f)[:-len("__plain.json")]; cond, proj = base.split("__")
        plain = load(f)
        for fx in ("seed", "iso"):
            g = f"{ROOT}/results/libfix/{base}__{fx}.json"
            if not os.path.exists(g): continue
            other = load(g); c = tot[fx]
            c["suites"] += 1; c["tests"] += len(plain)
            c["flaky_plain"] += sum(k == "flaky" for k in plain.values())
            c["flaky_fixture"] += sum(k == "flaky" for k in other.values())
            for t, k in plain.items():
                o = other.get(t)
                if k == "stable" and o == "always-fail": c["broken"] += 1; flaky_list.append((fx, cond, proj, t, "broken"))
                if k == "flaky" and o != "flaky": c["flaky_removed"] += 1
                if k == "stable" and o == "flaky": c["flaky_added"] += 1; flaky_list.append((fx, cond, proj, t, "made flaky"))
        for t, k in plain.items():
            if k == "flaky": flaky_list.append(("plain", cond, proj, t, "flaky"))
    out = {"by_fixture": {k: dict(v) for k, v in tot.items()}, "tests": flaky_list}
    json.dump(out, open(f"{ROOT}/results/library_fixtures_summary.json", "w"), indent=1)
    print(json.dumps(out["by_fixture"], indent=1))
    for x in flaky_list: print(x)


if __name__ == "__main__":
    main()
