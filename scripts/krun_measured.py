#!/usr/bin/env python3
"""Measured k run filters on the 200 run outcome sequences of results/matched_seq/ (Progress Report 3).

matched_run.py now records the outcome of every test in run order (one letter per run: P pass, F fail, E error,
S skip). Run i keeps the file order when i is even and shuffles when i is odd. A filter with k runs keeps a test only
if it passes all of its first k runs. For every flaky test (passed and failed in the 200 runs) of a plain or guard
suite we report whether it survives a 1, 3, 5, and 10 run filter, twice:
  file order:  the filter reruns the file in its written order (the even runs only), as a generate and filter
               pipeline would,
  shuffled:    the odd runs only (each in a new random order),
  mixed:       the first k runs as they happened (file order and shuffled alternating).
The M1 and M2 copies rerun the same tests and are left out. Writes results/krun_measured.json.

  python scripts/krun_measured.py                                   (results/matched_seq -> results/krun_measured.json)
  python scripts/krun_measured.py results/matched_heldout results/krun_heldout.json
"""
import glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS = (1, 3, 5, 10)


def survives(seq, k):
    return len(seq) >= k and all(c == "P" for c in seq[:k])


def main():
    rows = []
    src = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/matched_seq"
    for f in sorted(glob.glob(f"{src}/*.json")):
        j = json.load(open(f))
        for sid, s in j["suites"].items():
            if s["prompt"].endswith(("seed", "iso")):
                continue
            for t, v in s["tests"].items():
                seq = v.get("sequence", "")
                if not seq or "P" not in seq or not (set(seq) & {"F", "E"}):
                    continue
                ran = [c for c in seq if c in "PFE"]
                rows.append({"project": j["project"], "suite": sid, "model": s["model"], "prompt": s["prompt"], "test": t,
                             "fail_rate": round(sum(c in "FE" for c in ran) / len(ran), 3),
                             "file_order": {k: survives(seq[0::2], k) for k in KS},
                             "shuffled": {k: survives(seq[1::2], k) for k in KS},
                             "mixed": {k: survives(seq, k) for k in KS}})
    out = {"flaky_tests": len(rows), "rows": rows, "summary": {}}
    for mode in ("file_order", "shuffled", "mixed"):
        out["summary"][mode] = {k: sum(r[mode][k] for r in rows) for k in KS}
    for grp in sorted({(r["model"], r["prompt"]) for r in rows}):
        rs = [r for r in rows if (r["model"], r["prompt"]) == grp]
        out["summary"]["_".join(grp)] = {"flaky": len(rs), **{f"file_order_{k}": sum(r["file_order"][k] for r in rs) for k in KS}}
    json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/results/krun_measured.json", "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
