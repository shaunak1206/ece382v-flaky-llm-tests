#!/usr/bin/env python3
"""Static flakiness-risk scan of test functions (heuristic, per test function).

Categories (a test can have several):
  SLEEP           real sleep calls (time.sleep, asyncio.sleep with non zero literal)
  WALLCLOCK       reads the real clock (now, utcnow, today, time.time, monotonic, perf_counter)
                  without the clock being frozen or injected in that test
  TIMING_ASSERT   asserts on elapsed real time
  RANDOM          calls randomized APIs without a visible seed
  CONCURRENCY     threads, processes, executors, asyncio tasks
  FIXED_PATH      literal file paths outside tmp_path / tempfile
  SUBPROCESS      spawns a subprocess
  UNORDERED       asserts list(...) of a set / dict view against an ordered literal
"""
import ast, json, os, re, sys, glob

PAT = {
 "SLEEP": re.compile(r"\b(time\.)?sleep\(\s*(?!0\s*\))|asyncio\.sleep\(\s*(?!0\s*\))"),
 "WALLCLOCK": re.compile(r"\.(now|utcnow|today)\(|\btime\.(time|monotonic|perf_counter)\(|\b(perf_counter|monotonic)\(\)"),
 "TIMING_ASSERT": re.compile(r"(elapsed|duration|took|delta_t|t1\s*-\s*t0|end\s*-\s*start|time\.(time|monotonic|perf_counter)\(\)\s*-)"),
 "RANDOM": re.compile(r"\brandom\.(random|randint|randrange|choice|choices|shuffle|sample|uniform|getrandbits)\(|\brandom_(product|permutation|combination|combination_with_replacement|sample)\(|\bRRCache\(|\bwait_random(_exponential)?\(|\bsample\(|\bshuffle\("),
 "CONCURRENCY": re.compile(r"\bThread\(|ThreadPoolExecutor|ProcessPoolExecutor|multiprocessing|\bProcess\(|asyncio\.(gather|create_task|wait_for)|\.start\(\)\s*$", re.M),
 "FIXED_PATH": re.compile(r"""(open|Path|FileLock|SoftFileLock|JSONStorage|TinyDB)\(\s*f?["'](?!/tmp)[^"'{}]*\.(json|txt|lock|db|log|csv|dat)["']"""),
 "SUBPROCESS": re.compile(r"\bsubprocess\.|\bPopen\("),
 "UNORDERED": re.compile(r"assert\s+list\((?:[^()]|\([^()]*\))*(set\(|frozenset\(|\.keys\(\)|\.values\(\)|\.items\(\))(?:[^()]|\([^()]*\))*\)\s*==\s*\["),
}
CONTROL_CLOCK = re.compile(r"freeze_time|time_machine|travel\(|monkeypatch\.setattr\([^)]*time|timer\s*=|mock\.patch\([^)]*(time|datetime|now)|patch\.object\([^)]*(time|now)|fake_?(time|clock|timer)|FakeTime|frozen", re.I)
SEEDED = re.compile(r"\bseed\(|Random\(\s*\d|random_state|monkeypatch\.setattr\([^)]*random|mock\.patch\([^)]*(random|choice)|choice\s*=", re.I)
TMP = re.compile(r"tmp_path|tmpdir|tempfile|TemporaryDirectory|mkdtemp|NamedTemporaryFile")

def test_functions(path):
    try:
        src = open(path, encoding="utf8", errors="replace").read(); tree = ast.parse(src)
    except Exception:
        return []
    lines = src.splitlines(); mod_ctrl = bool(re.search(r"autouse\s*=\s*True[\s\S]{0,400}(freeze_time|time_machine)", src))
    mod_seed = bool(re.search(r"^random\.seed\(", src, re.M))
    out = []
    def visit(node, cls_dec=""):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                dec = "\n".join(ast.get_source_segment(src, d) or "" for d in ch.decorator_list)
                visit(ch, cls_dec + dec)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) and ch.name.startswith("test"):
                body = "\n".join(lines[ch.lineno - 1 - len(ch.decorator_list): ch.end_lineno])
                out.append((ch.name, body, cls_dec, mod_ctrl, mod_seed))
    visit(tree)
    return out

def scan(path):
    res = []
    for name, body, cls_dec, mod_ctrl, mod_seed in test_functions(path):
        ctx = body + "\n" + cls_dec
        tags = set()
        for k, p in PAT.items():
            if p.search(body): tags.add(k)
        if "WALLCLOCK" in tags and (CONTROL_CLOCK.search(ctx) or mod_ctrl): tags.discard("WALLCLOCK")
        if "TIMING_ASSERT" in tags and not ("assert" in body and "WALLCLOCK" in tags | {"x"} and re.search(r"time\.(time|monotonic|perf_counter)|perf_counter|monotonic", body)): tags.discard("TIMING_ASSERT")
        if "RANDOM" in tags and (SEEDED.search(ctx) or mod_seed): tags.discard("RANDOM")
        if "FIXED_PATH" in tags and TMP.search(body): tags.discard("FIXED_PATH")
        res.append({"file": path, "test": name, "smells": sorted(tags)})
    return res

def scan_suite(files):
    tests = [t for f in files for t in scan(f)]
    cnt = {k: sum(1 for t in tests if k in t["smells"]) for k in PAT}
    anyc = sum(1 for t in tests if t["smells"])
    return {"n_tests": len(tests), "any": anyc, "counts": cnt, "tests": tests}

if __name__ == "__main__":
    out = scan_suite(sys.argv[2:]); print(json.dumps({k: v for k, v in out.items() if k != "tests"}))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
