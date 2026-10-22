#!/usr/bin/env python3
"""Run the static smell scan over every suite and write results/smells.json plus a markdown table."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from smells import scan_suite, PAT

R = "/Users/shaunkapoor/ece382v-flaky-llm-tests"
HUMAN = {
 "cachetools": "subjects/cachetools/tests/*.py", "schedule": "subjects/schedule/test_schedule.py",
 "tinydb": "subjects/tinydb/tests/*.py", "humanize": "subjects/humanize/tests/test_[!b]*.py",
 "toolz": "subjects/toolz/toolz/tests/*.py", "tabulate": "subjects/python-tabulate/test/*.py",
 "tenacity": "subjects/tenacity/tests/**/*.py", "filelock": "subjects/filelock/tests/*.py",
 "boltons": "subjects/boltons/tests/*.py", "more_itertools": "subjects/more-itertools/tests/*.py",
 "arrow": "subjects/arrow/tests/*.py", "pybreaker": "subjects/pybreaker/tests/pybreaker_test.py",
}
CONDS = {"human": None, "llm_tests": "AG-S", "gen_oneshot_sonnet": "OS-S", "gen_oneshot_haiku": "OS-H",
         "gen_tp_sonnet": "TP-S", "gen_plain_sonnet": "PL-S"}


def files(cond, proj):
    if cond == "human":
        return sorted(f for f in glob.glob(f"{R}/{HUMAN[proj]}", recursive=True) if os.path.basename(f).startswith("test") or f.endswith("_test.py"))
    return sorted(glob.glob(f"{R}/{cond}/{proj}/test_*.py"))


def main():
    out = {}
    for cond in CONDS:
        for proj in HUMAN:
            fs = files(cond, proj)
            if not fs:
                continue
            s = scan_suite(fs)
            out[f"{cond}/{proj}"] = s
    json.dump(out, open(f"{R}/results/smells.json", "w"), indent=1)
    print("| Condition | Suites | Test functions | Any smell % | " + " | ".join(PAT) + " |")
    print("|---|---|---|---|" + "---|" * len(PAT))
    for cond, lab in CONDS.items():
        ss = [v for k, v in out.items() if k.startswith(cond + "/")]
        n = sum(s["n_tests"] for s in ss)
        if not n:
            continue
        anyc = sum(s["any"] for s in ss)
        cells = [f"{100 * sum(s['counts'][k] for s in ss) / n:.1f}" for k in PAT]
        print(f"| {lab or 'Human'} | {len(ss)} | {n} | {100 * anyc / n:.1f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
