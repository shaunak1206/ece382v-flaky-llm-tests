#!/usr/bin/env python3
"""RQ3 on the known flaky code study: which features of a generated test function go with flipping.

For every generated suite that ran 200 times (results/matched, matched_guard, matched_guard3, matched_mitig guard2,
matched_heldout, matched_guard3cl, matched_guard3gpt, matched_od; M1 and M2 copies are left out because they rerun
the same tests), parse test_gen.py and mark each test function with the features below. A function is kept if any of
its cases passed the first run, and flaky if any kept case both passed and failed. Writes
results/feature_flakiness_matched.json with, per feature, the kept functions that use it, how many of those flipped,
and the same for the kept functions that do not use it.

The features are static and name based (a call to random.random, a read of time.time, a write under /tmp), so a
helper that hides a feature behind another function is missed. Seeding counts as present when the function, or the
module at top level, calls a seed function.

  python scripts/feature_flakiness_matched.py
"""
import ast, collections, glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ("matched", "matched_guard", "matched_guard3", "matched_mitig", "matched_heldout", "matched_guard3cl",
        "matched_guard3gpt", "matched_od")
RANDOM = re.compile(r"^(random\.|np\.random\.|numpy\.random\.|torch\.(rand|randn|randint|randperm|normal|bernoulli|multinomial))")
SEED = re.compile(r"(^|\.)(seed|manual_seed|default_rng|RandomState|Random)$")
CLOCK = re.compile(r"^(time\.(time|monotonic|perf_counter|process_time)|datetime\.(datetime\.)?(now|utcnow|today)|date\.today)$")
FS_WRITE = re.compile(r"^(os\.(mkdir|makedirs|remove|unlink|rename|rmdir|chdir)|shutil\.(rmtree|copy|copyfile|copytree|move)|"
                      r"tempfile\.(mkdtemp|mkstemp|NamedTemporaryFile|TemporaryDirectory)|pathlib\.Path\.(write_text|write_bytes|mkdir))$")
FEATURES = {
    "random": "Calls a random generator",
    "random_unseeded": "  without a seed in the test or module",
    "clock": "Reads the real clock",
    "sleep": "Sleeps for real",
    "concurrency": "Threads, processes, or asyncio",
    "module_state": "Reloads a module or edits sys.modules",
    "filesystem": "Writes files outside tmp_path",
    "hash_or_id": "Calls hash() or id()",
    "monkeypatch": "Uses monkeypatch or mock.patch",
    "freeze": "Freezes the clock",
}


def dotted(n):
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr); n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    return ".".join(reversed(parts))


def features(fn, module_seeded, src):
    calls = [dotted(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    text = ast.get_source_segment(src, fn) or ""
    args = {a.arg for a in fn.args.args}
    f = set()
    if any(RANDOM.match(c) for c in calls):
        f.add("random")
        if not (module_seeded or any(SEED.search(c) for c in calls)):
            f.add("random_unseeded")
    if any(CLOCK.match(c) for c in calls): f.add("clock")
    if any(c in ("time.sleep", "asyncio.sleep") for c in calls) and "freeze" not in text: f.add("sleep")
    if re.search(r"\b(threading|multiprocessing|concurrent\.futures|asyncio)\b", text) or "async def" in text: f.add("concurrency")
    if re.search(r"importlib\.reload|sys\.modules|reload\(", text): f.add("module_state")
    writes_open = any(isinstance(n, ast.Call) and dotted(n.func) == "open" and any(
        isinstance(a, ast.Constant) and isinstance(a.value, str) and set(a.value) & set("wax") for a in n.args[1:2] + [k.value for k in n.keywords if k.arg == "mode"])
        for n in ast.walk(fn))
    if ("tmp_path" not in args and "tmpdir" not in args) and (writes_open or any(FS_WRITE.match(c) for c in calls) or "/tmp" in text):
        f.add("filesystem")
    if any(c in ("hash", "id") for c in calls): f.add("hash_or_id")
    if "monkeypatch" in args or re.search(r"mock\.patch|patch\(|patch\.object", text): f.add("monkeypatch")
    if re.search(r"freeze_time|freezegun|time_machine", text): f.add("freeze")
    return f


def main():
    use = collections.Counter(); flaky = collections.Counter(); total = flaky_total = 0
    s_use = collections.Counter(); s_flaky = collections.Counter(); s_total = s_flaky_total = 0; none_flaky = 0
    per_suite_seen = set()
    for d in DIRS:
        for path in sorted(glob.glob(f"{ROOT}/results/{d}/*.json")):
            j = json.load(open(path)); p = j["project"]
            for sid, s in j["suites"].items():
                if s.get("prompt", "").endswith(("seed", "iso")) or (p, sid) in per_suite_seen: continue
                per_suite_seen.add((p, sid))
                f = f"{ROOT}/matched/{p}/{sid}/test_gen.py"
                if not os.path.exists(f) or all(t.startswith("::") for t in s["tests"]): continue
                src = open(f, encoding="utf-8", errors="replace").read()
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    continue
                module_seeded = any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and SEED.search(dotted(n.value.func))
                                    for n in tree.body)
                first = set(s["first_run_failed"])
                status = {}  # function name -> (kept, flaky)
                for t, v in s["tests"].items():
                    name = t.split("::")[-1].split("[")[0]
                    k, fl = status.get(name, (False, False))
                    kept = t not in first
                    status[name] = (k or kept, fl or (kept and v["kind"] == "flaky"))
                suite_f = set(); suite_fl = False
                for fn in ast.walk(tree):
                    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.name.startswith("test"): continue
                    kept, fl = status.get(fn.name, (False, False))
                    feats = features(fn, module_seeded, src)
                    suite_f |= feats  # a polluter can be a removed test, so the suite view uses every function
                    if not kept: continue
                    total += 1; flaky_total += fl; suite_fl |= fl
                    none_flaky += fl and not feats
                    for x in feats:
                        use[x] += 1; flaky[x] += fl
                s_total += 1; s_flaky_total += suite_fl
                for x in suite_f:
                    s_use[x] += 1; s_flaky[x] += suite_fl
    out = {"kept_functions": total, "flaky_functions": flaky_total, "flaky_functions_with_no_feature": none_flaky,
           "suites": s_total, "suites_with_flaky": s_flaky_total, "features": {}}
    for x, label in FEATURES.items():
        out["features"][x] = {"label": label.strip(), "use": use[x], "share": round(100 * use[x] / total, 1),
                              "flaky": flaky[x], "flaky_rate": round(100 * flaky[x] / use[x], 2) if use[x] else None,
                              "without": total - use[x], "flaky_without": flaky_total - flaky[x],
                              "flaky_rate_without": round(100 * (flaky_total - flaky[x]) / (total - use[x]), 2),
                              "suites_use": s_use[x], "suites_use_flaky": s_flaky[x],
                              "suite_flaky_rate": round(100 * s_flaky[x] / s_use[x], 1) if s_use[x] else None,
                              "suite_flaky_rate_without": round(100 * (s_flaky_total - s_flaky[x]) / (s_total - s_use[x]), 1)}
    json.dump(out, open(f"{ROOT}/results/feature_flakiness_matched.json", "w"), indent=1)
    print(total, "kept functions,", flaky_total, "flaky,", none_flaky, "flaky with no feature;", s_total, "suites,", s_flaky_total, "with a kept flaky test")
    for x, v in out["features"].items():
        print(f'{v["label"]:45s} use {v["use"]:6d} ({v["share"]:5.1f}%) flaky {v["flaky"]:3d} rate {v["flaky_rate"]}%  without {v["flaky_rate_without"]}% | suites {v["suites_use"]} flaky {v["suites_use_flaky"]} ({v["suite_flaky_rate"]}% vs {v["suite_flaky_rate_without"]}%)')


if __name__ == "__main__":
    main()
