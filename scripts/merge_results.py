#!/usr/bin/env python3
"""Merge matched_run.py results from several CI runs into one folder, one JSON per project.

Each run of the matched-study workflow writes <project>.json with the suites it ran. Runs that split the suites
(for example held out samples 1 to 4 and 5 to 8) are merged here: suites are added to the project's file, and the
human control of each later run is kept under merged_human.

  python scripts/merge_results.py <downloaded artifact folder> results/matched_heldout
"""
import glob, json, os, sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{src}/*.json")):
        j = json.load(open(f))
        out = f"{dst}/{os.path.basename(f)}"
        if os.path.exists(out):
            o = json.load(open(out))
            new = {k: v for k, v in j["suites"].items() if k not in o["suites"]}
            if not new: continue  # already merged
            o["suites"].update(new); o.setdefault("merged_human", []).append(j.get("human"))
            j = o
        json.dump(j, open(out, "w"), indent=1)
        n += 1
    print(n, "files updated in", dst)


if __name__ == "__main__":
    main()
