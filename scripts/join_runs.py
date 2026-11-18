#!/usr/bin/env python3
"""Join matched_run.py results of one project whose runs were split across CI jobs (--start).

A suite that is too slow for one 6 hour CI job (for example one whose tests hit the 60 second timeout in every run)
runs as several jobs over disjoint run ranges. Run i always uses seed 7000+i and the same order rule, so the joined
result is the same 200 runs a single job would have made. This rebuilds each suite's per run sequence and recomputes
the counts, the kind, and the filter run (run 0) from it.

  python scripts/join_runs.py out.json chunk_0_99.json chunk_100_199.json
"""
import json, sys


def join(chunks):
    base = json.loads(json.dumps(min(chunks, key=lambda c: min(s.get("start", 0) for s in c["suites"].values()))))
    for sid in {k for c in chunks for k in c["suites"]}:
        parts = sorted((c["suites"][sid] for c in chunks if sid in c["suites"]), key=lambda s: s.get("start", 0))
        total = max(p.get("start", 0) + p["runs"] for p in parts)
        names = {t for p in parts for t in p["tests"]}
        tests = {}
        for t in names:
            seq = ["-"] * total; msg = None
            for p in parts:
                v = p["tests"].get(t)
                if not v: continue
                msg = msg or v.get("first_message")
                for i, ch in enumerate(v["sequence"]):
                    if ch != "-": seq[i] = ch
            c = {x: seq.count(x) for x in "PFES"}
            p_, f_ = c["P"], c["F"] + c["E"]
            kind = "flaky" if p_ and f_ else "always-fail" if f_ else "always-skip" if not p_ else "stable"
            tests[t] = {"pass": p_, "fail": c["F"], "error": c["E"], "skip": c["S"], "kind": kind,
                        "fails_file_order": sum(seq[i] in "FE" for i in range(0, total, 2)),
                        "fails_shuffled": sum(seq[i] in "FE" for i in range(1, total, 2)),
                        "sequence": "".join(seq), "first_message": msg}
        first = parts[0]
        assert first.get("start", 0) == 0, f"{sid}: no chunk starts at run 0"
        base["suites"][sid] = {"model": first["model"], "prompt": first["prompt"], "sample": first["sample"],
                               "runs": total, "joined_chunks": [[p.get("start", 0), p.get("start", 0) + p["runs"]] for p in parts],
                               "n_tests": len(tests), "first_run_failed": first["first_run_failed"], "tests": tests,
                               "flaky": sorted(t for t, v in tests.items() if v["kind"] == "flaky"),
                               "always_fail": sorted(t for t, v in tests.items() if v["kind"] == "always-fail")}
    base["seconds"] = sum(c.get("seconds", 0) for c in chunks)
    return base


def main():
    out, *ins = sys.argv[1:]
    j = join([json.load(open(f)) for f in ins])
    json.dump(j, open(out, "w"), indent=1)
    for sid, s in sorted(j["suites"].items()):
        print(sid, s["runs"], s["joined_chunks"], "flaky", len(s["flaky"]))


if __name__ == "__main__":
    main()
