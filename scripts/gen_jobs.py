#!/usr/bin/env python3
"""Write the generation job list for Progress Report 3 and run it with a spend cap.

Jobs (one line each: project model sample variant), for the given models:
  held out sample:        every project of results/gruber/matched_projects.txt, samples 1 to 8, variant heldout
  guard v3:               the same projects, samples 1 and 2, variant guard3
  order dependent arm:    every project of results/gruber/matched_projects_od.txt, samples 1 to 4, variant plain
A job whose suite file already exists is skipped, so the list can be rerun after a stop.

  python scripts/gen_jobs.py --models gpt-5.4-mini,gpt-4.1-mini --out /tmp/jobs.txt
  python scripts/gen_jobs.py --run /tmp/jobs.txt --workers 8 --cap 18     (OpenAI only, stops near the cap)
The spend estimate reads results/openai_calls.jsonl from line --since (default 291, the first held out GPT call)
with conservative prices, so it overstates the bill.
"""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = {"gpt-4.1-mini": "gpt41mini", "gpt-5.4-mini": "gpt54mini", "claude-sonnet-5-5": "sonnet", "claude-haiku-5-5": "haiku"}
PRICE = {"gpt-4.1-mini": (0.40, 1.60), "gpt-5.4-mini": (1.00, 6.00)}  # $ per million tokens, in and out, high on purpose


def jobs(models):
    main = open(f"{ROOT}/results/gruber/matched_projects.txt").read().split()
    od = open(f"{ROOT}/results/gruber/matched_projects_od.txt").read().split()
    out = []
    for k in range(1, 9):
        out += [(p, m, k, "heldout") for p in main for m in models]
    for k in (1, 2):
        out += [(p, m, k, "guard3") for p in main for m in models]
    for k in range(1, 5):
        out += [(p, m, k, "plain") for p in od for m in models]
    return out


def done(p, m, k, v):
    f = f"{ROOT}/matched/{p}/{TAGS[m]}_{v}_{k}/test_gen.py"
    return os.path.exists(f) and os.path.getsize(f) > 0


def spend(since):
    t = 0.0
    with open(f"{ROOT}/results/openai_calls.jsonl") as fh:
        for i, line in enumerate(fh, 1):
            if i < since: continue
            j = json.loads(line); u = j["usage"]; pi, po = PRICE.get(j["model"], (2.0, 8.0))
            t += u.get("prompt_tokens", 0) * pi / 1e6 + u.get("completion_tokens", 0) * po / 1e6
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.4-mini,gpt-4.1-mini"); ap.add_argument("--out")
    ap.add_argument("--run"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=float, default=18.0); ap.add_argument("--since", type=int, default=291)
    a = ap.parse_args()
    if a.out:
        js = jobs(a.models.split(","))
        open(a.out, "w").write("".join(f"{p} {m} {k} {v}\n" for p, m, k, v in js))
        print(len(js), "jobs,", sum(not done(*j) for j in js), "to do")
    if a.run:
        js = [l.split() for l in open(a.run) if l.strip()]
        js = [(p, m, int(k), v) for p, m, k, v in js if not done(p, m, int(k), v)]
        stop = threading.Event(); lock = threading.Lock()
        print(f"{len(js)} jobs, spend so far ${spend(a.since):.2f}, cap ${a.cap}", flush=True)

        def one(j):
            p, m, k, v = j
            for attempt in range(3):
                if stop.is_set() or done(*j): return
                with lock:
                    s = spend(a.since)
                    if s > a.cap:
                        stop.set(); print(f"STOP: estimated spend ${s:.2f} > cap", flush=True); return
                r = subprocess.run([sys.executable, f"{ROOT}/scripts/gen_openai.py", "matched", p, m, str(k), v],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    print(r.stdout.strip(), flush=True); return
                print(f"retry {attempt + 1} {p} {m} {k} {v}: {r.stderr.strip()[-200:]}", flush=True)
                time.sleep(20 * (attempt + 1))
            print(f"FAILED {p} {m} {k} {v}", flush=True)

        with cf.ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, js))
        print(f"finished, estimated spend ${spend(a.since):.2f}", flush=True)


if __name__ == "__main__":
    main()
