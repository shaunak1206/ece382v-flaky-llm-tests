#!/usr/bin/env python3
"""Per test function features that plausibly relate to flakiness (AST based, stdlib only).

Walks every test function of every suite (Human, AG-S, OS-S, OS-H, TP-S, PL-S; the same file
sets as scripts/run_smells.py) and records, per test function:

  shared state   module level objects read or written, class attributes, library singletons and
                 registries (schedule default scheduler, filelock singletons, arrow locale
                 registry, humanize i18n, library module flags), os.environ / locale / sys / cwd
                 mutation, setup without teardown, non function scoped fixtures
  isolation      tmp_path / tmpdir / tempfile, monkeypatch, freezegun, time_machine, mock.patch,
                 fresh library objects built in the test or in a function scoped fixture
  time           real clock reads, real sleeps, timing assertions with threshold, planned delay
                 and slack (see TIMING below), clock derived time of day strings
  randomness     random calls, seeded or not, assertions on random values exact or property
  concurrency    threads, processes, asyncio, joins, joins with timeouts
  size, style    asserts, lines, parametrize cases, classes, unittest style

Fixture features are merged into the test when the test requests the fixture (by argument,
usefixtures, or autouse in the same file, class, or a conftest.py on the path).

TIMING. A timing assertion is a comparison in an assert (or self.assertLess etc.) where one side
is a measured elapsed time: a name assigned from `clock() - x`, an inline `clock() - x`, or a
tenacity statistics key `delay_since_first_attempt`. For each one we record direction (upper:
elapsed < T, lower: elapsed >= T), the threshold T when it is a constant expression, and the
planned delay inside the measured window (from `start = clock()` to the measurement) as
[min, nominal, max] seconds. Planned delay sources: literal sleeps (times literal loop counts),
tenacity Retrying/@retry configs (simulated with the library's stop/wait semantics), filelock
acquire timeouts that are expected to raise Timeout, more_itertools.time_limited limits. Slack is
  upper bound: T - planned_max      (how much scheduling overhead the test tolerates)
  lower bound: planned_min - T      (negative means the test can fail on an idle machine)
Cases the model cannot derive are listed in MANUAL_DELAY with the hand derivation.

Writes results/characterize.json and prints the per condition table.
Usage: .venv/bin/python scripts/characterize.py
"""
import ast, glob, json, math, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_smells import HUMAN, files as suite_files  # noqa: E402

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = {"human": "Human", "llm_tests": "AG-S", "gen_oneshot_sonnet": "OS-S",
         "gen_oneshot_haiku": "OS-H", "gen_tp_sonnet": "TP-S", "gen_plain_sonnet": "PL-S",
         "gen_plain_gpt54mini": "PL-G5", "gen_plain_gpt41mini": "PL-G4"}
LIBS = list(HUMAN)  # import name == project key for all 12

CLOCK = {"time.time", "time.monotonic", "time.perf_counter", "time.time_ns", "time.monotonic_ns",
         "time.perf_counter_ns", "datetime.datetime.now", "datetime.datetime.utcnow",
         "datetime.datetime.today", "datetime.date.today", "arrow.now", "arrow.utcnow",
         "arrow.Arrow.now", "arrow.Arrow.utcnow", "arrow.arrow.Arrow.now", "arrow.arrow.Arrow.utcnow"}
ELAPSED_CLOCK = {"time.time", "time.monotonic", "time.perf_counter", "time.time_ns",
                 "time.monotonic_ns", "time.perf_counter_ns"}
SLEEP = {"time.sleep", "asyncio.sleep", "tenacity.sleep", "trio.sleep", "anyio.sleep"}
RANDOM_STD = {"random." + f for f in ("random", "randint", "randrange", "choice", "choices", "shuffle",
              "sample", "uniform", "getrandbits", "gauss", "triangular", "betavariate", "expovariate",
              "normalvariate", "randbytes")}
RANDOM_LIB = {"random_product", "random_permutation", "random_combination",
              "random_combination_with_replacement", "sample", "random_sample", "RRCache",
              "wait_random", "wait_random_exponential", "wait_exponential_jitter", "wait_full_jitter"}
PROPERTY_WRAP = {"sorted", "set", "frozenset", "len", "Counter", "sum", "min", "max", "isinstance",
                 "type", "all", "any", "bool", "tuple_sorted"}
MUTATING = {"append", "extend", "add", "update", "clear", "pop", "popitem", "insert", "remove",
            "discard", "setdefault", "register", "__setitem__", "__delitem__", "appendleft"}
ISO_FIXTURES = {"tmp_path": "tmp_path", "tmpdir": "tmp_path", "tmp_path_factory": "tmp_path",
                "tmpdir_factory": "tmp_path", "monkeypatch": "monkeypatch", "freezer": "freezegun",
                "time_machine": "time_machine", "mocker": "mock_patch"}
SCHEDULE_DEFAULT = {"every", "run_pending", "run_all", "clear", "cancel_job", "get_jobs", "jobs",
                    "next_run", "idle_seconds", "default_scheduler", "repeat", "get_next_run"}
LIB_FLAGS = {"tabulate": {"PRESERVE_WHITESPACE", "MIN_PADDING", "WIDE_CHARS_MODE", "_table_formats",
                          "multiline_formats", "tabulate_formats"},
             "tinydb": {"default_table_name", "default_storage_class", "default_query_cache_capacity",
                        "table_class", "document_class", "document_id_class"}}

# Library value types that cannot be mutated, so a module level instance is not shared state.
IMMUTABLE_LIB = {"arrow.Arrow", "arrow.arrow.Arrow", "arrow.get", "tinydb.Query", "tinydb.queries.Query",
                 "tinydb.where"}

# Hand derived planned delays (seconds) for timing windows the model cannot simulate.
# key: (cond, proj, test name) -> (min, nominal, max, derivation)
# key: (cond, proj, test, assert line, direction) -> (threshold override or None, min, nominal, max, derivation)
MANUAL_DELAY = {
    ("human", "tenacity", "test_with_wait", 1343, "lower"):
        (None, 0.25, 0.25, 0.25, "module level @retry wait_fixed(0.05), NoneReturnUntilAfterCount(5) gives 5 waits"),
    ("human", "tenacity", "test_redefine_wait", 1784, "lower"):
        (None, 0.5, 0.5, 0.5, "retry_with(wait_fixed(0.1)), NoneReturnUntilAfterCount(5) gives 5 waits"),
    ("human", "boltons", "test_wraps_async", 263, "lower"):
        (None, 0.3, 0.3, 0.3, "asyncio.run(foo(0.3)) where foo awaits asyncio.sleep(x)"),
    ("gen_oneshot_sonnet", "tenacity", "test_wait_strategies_sleep_for_the_expected_durations", 105, "lower"):
        (-0.01, 0.0, 0.0, 0.0, "threshold is sum(expected) - 0.01 and the planned delay is sum(expected); shown relative to it"),
    ("gen_oneshot_sonnet", "tenacity", "test_wait_strategies_sleep_for_the_expected_durations", 106, "upper"):
        (1.0, 0.0, 0.0, 0.0, "threshold is sum(expected) + 1.0; shown relative to the planned sum(expected)"),
    ("gen_oneshot_sonnet", "tenacity", "test_wait_random_sleeps_stay_within_bounds", 154, "lower"):
        (-0.01, 0.0, 0.0, 0.0, "threshold is sum(recorder.sleeps) - 0.01, the sleeps actually taken; shown relative to them"),
    ("gen_oneshot_sonnet", "tenacity", "test_sleep_helpers_block_for_about_the_requested_time", 205, "lower"):
        (None, 0.1, 0.1, 0.1, "sleep_using_event(never_set)(0.1) waits the full 0.1 s on an unset Event"),
    ("gen_oneshot_sonnet", "tenacity", "test_sleep_helpers_block_for_about_the_requested_time", 206, "upper"):
        (None, 0.1, 0.1, 0.1, "sleep_using_event(never_set)(0.1) waits the full 0.1 s on an unset Event"),
    ("gen_oneshot_sonnet", "tenacity", "test_event_can_interrupt_a_long_wait", 231, "upper"):
        (None, 0.1, 0.1, 0.1, "threading.Timer(0.1, event.set) interrupts wait_fixed(5) after 0.1 s"),
}


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr); node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id); return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        inner = dotted(node.func)
        if inner:
            parts.append(inner + "()"); return ".".join(reversed(parts))
    return None


class Module:
    def __init__(self, path, lib, extra_fixture_files=()):
        self.path, self.lib = path, lib
        self.src = open(path, encoding="utf8", errors="replace").read()
        self.tree = ast.parse(self.src)
        self.alias = {}          # local name -> fully qualified
        self.mod_state = {}      # module level name -> kind
        self.mod_numbers = {}
        self.fixtures = {}       # name -> FunctionDef node (module level)
        self.mod_setup = set()
        for node in self.tree.body:
            self._top(node)
        for f in extra_fixture_files:   # conftest.py fixtures
            try:
                t = ast.parse(open(f, encoding="utf8", errors="replace").read())
            except SyntaxError:
                continue
            cm = Module.__new__(Module); cm.path, cm.lib, cm.alias, cm.mod_state, cm.mod_numbers = f, lib, {}, {}, {}
            cm.fixtures, cm.mod_setup, cm.tree = {}, set(), t
            cm.src = open(f, encoding="utf8", errors="replace").read()
            for node in t.body:
                cm._top(node)
            for k, v in cm.fixtures.items():
                self.fixtures.setdefault(k, (v, cm))
        self.fixtures = {k: (v if isinstance(v, tuple) else (v, self)) for k, v in self.fixtures.items()}

    def resolve(self, name):
        if not name:
            return name
        head, _, rest = name.partition(".")
        full = self.alias.get(head, head)
        return full + ("." + rest if rest else "")

    def is_lib(self, fq):
        return fq is not None and (fq == self.lib or fq.startswith(self.lib + "."))

    def _top(self, node):
        if isinstance(node, ast.Import):
            for a in node.names:
                self.alias[(a.asname or a.name).split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                self.alias[a.asname or a.name] = f"{node.module}.{a.name}"
        elif isinstance(node, (ast.Try, ast.If)):
            for ch in node.body + getattr(node, "orelse", []) + [h for x in getattr(node, "handlers", []) for h in x.body]:
                self._top(ch)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            val = node.value
            if val is None:
                return
            kind = self._kind(val)
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and n.id != "pytestmark":
                        self.mod_state[n.id] = kind
                        v = num(val, self.mod_numbers)
                        if v is not None:
                            self.mod_numbers[n.id] = v
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if fixture_info(node, self) is not None:
                self.fixtures[node.name] = node
            elif node.name in ("setup_module", "setup_function", "setUpModule"):
                self.mod_setup.add(node.name)
            elif node.name in ("teardown_module", "teardown_function", "tearDownModule"):
                self.mod_setup.add(node.name)
            elif not node.name.startswith("test") and node.decorator_list:
                for d in node.decorator_list:
                    fq = self.resolve(dotted(d.func if isinstance(d, ast.Call) else d))
                    if self.is_lib(fq):
                        self.mod_state[node.name] = "lib_decorated_func"

    def _kind(self, val):
        if isinstance(val, ast.Call):
            fq = self.resolve(dotted(val.func))
            if fq in IMMUTABLE_LIB or (fq or "").split(".")[-1] in ("Query",):
                return "constant"
            if self.is_lib(fq) and fq.split(".")[-1][:1].isupper():
                return "lib_instance"
            return "call_result"
        if isinstance(val, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
            return "mutable_literal"
        if isinstance(val, (ast.Lambda,)):
            return "constant"
        return "constant"


def fixture_info(fn, mod):
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        name = dotted(target) or ""
        if name.endswith("fixture") or mod.resolve(name) in ("pytest.fixture", "pytest_asyncio.fixture"):
            scope, autouse, params = "function", False, None
            if isinstance(d, ast.Call):
                for k in d.keywords:
                    if k.arg == "scope" and isinstance(k.value, ast.Constant):
                        scope = k.value.value
                    if k.arg == "autouse" and isinstance(k.value, ast.Constant):
                        autouse = bool(k.value.value)
                    if k.arg == "params" and isinstance(k.value, (ast.List, ast.Tuple)):
                        params = len(k.value.elts)
            return {"scope": scope, "autouse": autouse, "params": params,
                    "yields": any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn)),
                    "args": [a.arg for a in fn.args.args if a.arg not in ("self", "cls")]}
    return None


def num(node, env=None):
    env = env or {}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = num(node.operand, env); return None if v is None else -v
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp):
        a, b = num(node.left, env), num(node.right, env)
        if a is None or b is None:
            return None
        try:
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b,
                    ast.Div: a / b if b else None, ast.Pow: a ** b}.get(type(node.op))
        except Exception:
            return None
    if isinstance(node, ast.Call):
        fn = dotted(node.func) or ""
        if fn.endswith("timedelta"):
            unit = {"days": 86400, "hours": 3600, "minutes": 60, "seconds": 1, "milliseconds": 1e-3,
                    "microseconds": 1e-6}
            tot = 0.0
            for k in node.keywords:
                v = num(k.value, env)
                if v is None or k.arg not in unit:
                    return None
                tot += v * unit[k.arg]
            if node.args:
                v = num(node.args[0], env)
                if v is None:
                    return None
                tot += v * 86400
            return tot
        if fn.endswith("total_seconds") and isinstance(node.func, ast.Attribute):
            return num(node.func.value, env)
    return None


def kw(call, name, pos=None, default=None, env=None):
    for k in call.keywords:
        if k.arg == name:
            v = num(k.value, env); return default if v is None else v
    if pos is not None and len(call.args) > pos:
        v = num(call.args[pos], env); return default if v is None else v
    return default


# ---------------------------------------------------------------- tenacity planned delay model
def tenacity_waits(wnode, mod, env):
    """Return function attempt -> (min, nom, max) seconds, or None."""
    if wnode is None:
        return lambda a: (0.0, 0.0, 0.0)
    if not isinstance(wnode, ast.Call):
        return None
    n = (mod.resolve(dotted(wnode.func)) or "").split(".")[-1]
    if n == "wait_fixed":
        w = kw(wnode, "wait", 0, env=env)
        return None if w is None else (lambda a: (w, w, w))
    if n == "wait_none":
        return lambda a: (0.0, 0.0, 0.0)
    if n == "wait_random":
        lo, hi = kw(wnode, "min", 0, 0.0, env), kw(wnode, "max", 1, 1.0, env)
        return lambda a: (lo, (lo + hi) / 2, hi)
    if n == "wait_incrementing":
        st, inc, mx = kw(wnode, "start", 0, 0.0, env), kw(wnode, "increment", 1, 100.0, env), kw(wnode, "max", 2, math.inf, env)
        return lambda a: (max(0, min(mx, st + inc * (a - 1))),) * 3
    if n in ("wait_exponential", "wait_random_exponential", "wait_full_jitter"):
        m, mx = kw(wnode, "multiplier", 0, 1.0, env), kw(wnode, "max", 1, math.inf, env)
        eb, mn = kw(wnode, "exp_base", 2, 2.0, env), kw(wnode, "min", 3, 0.0, env)
        def hi(a):
            return max(max(0, mn), min(m * eb ** (a - 1), mx))
        if n == "wait_exponential":
            return lambda a: (hi(a),) * 3
        return lambda a: (mn, (mn + hi(a)) / 2, hi(a))
    if n == "wait_exponential_jitter":
        ini, mx = kw(wnode, "initial", 0, 1.0, env), kw(wnode, "max", 1, math.inf, env)
        eb, jit = kw(wnode, "exp_base", 2, 2.0, env), kw(wnode, "jitter", 3, 1.0, env)
        return lambda a: (min(mx, ini * eb ** (a - 1)), min(mx, ini * eb ** (a - 1) + jit / 2),
                          min(mx, ini * eb ** (a - 1) + jit))
    if n == "wait_chain":
        subs = [tenacity_waits(x, mod, env) for x in wnode.args]
        if not subs or any(s is None for s in subs):
            return None
        return lambda a: subs[min(a, len(subs)) - 1](1)
    return None


def tenacity_stop(snode, mod, env):
    """Return a predicate stop(attempt, t_since_start, upcoming_sleep) -> bool, ("never",) or None."""
    if snode is None:
        return "never"
    if isinstance(snode, ast.BinOp) and isinstance(snode.op, (ast.BitOr, ast.BitAnd)):
        a, b = tenacity_stop(snode.left, mod, env), tenacity_stop(snode.right, mod, env)
        if a is None or b is None or "never" in (a, b):
            return None
        return (lambda *x: a(*x) or b(*x)) if isinstance(snode.op, ast.BitOr) else (lambda *x: a(*x) and b(*x))
    if not isinstance(snode, ast.Call):
        n = (mod.resolve(dotted(snode)) or "").split(".")[-1]
        return "never" if n == "stop_never" else None
    n = (mod.resolve(dotted(snode.func)) or "").split(".")[-1]
    if n == "stop_after_attempt":
        v = kw(snode, "max_attempt_number", 0, env=env)
        return None if v is None else (lambda a, t, w: a >= v)
    if n == "stop_after_delay":
        v = kw(snode, "max_delay", 0, env=env)
        return None if v is None else (lambda a, t, w: t >= v)
    if n == "stop_before_delay":
        v = kw(snode, "max_delay", 0, env=env)
        return None if v is None else (lambda a, t, w: t + w >= v)
    if n in ("stop_any", "stop_all"):
        subs = [tenacity_stop(x, mod, env) for x in snode.args]
        if not subs or any(x is None or x == "never" for x in subs):
            return None
        return (lambda *x: any(f(*x) for f in subs)) if n == "stop_any" else (lambda *x: all(f(*x) for f in subs))
    return None


def simulate(waits, stop, failures):
    """Total sleep time of a retry loop with zero call overhead: [min, nominal, max] seconds."""
    if stop == "never" and failures is None:
        return None
    out = []
    for i in range(3):
        t, a = 0.0, 1
        while a < 10000:
            if failures is not None and a > failures:
                break
            w = waits(a)[i]
            if stop != "never" and stop(a, t, w):
                break
            t += w; a += 1
        out.append(round(t, 6))
    return out


def infer_failures(fn_node, mod):
    """Number of failing calls before success, None = always fails, 'unknown'."""
    src = ast.unparse(fn_node)
    if re.search(r"raises\(\s*(tenacity\.)?RetryError|except\s*\(?\s*(tenacity\.)?RetryError", src):
        return None
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id[:1].isupper() \
                and "Flaky" in n.func.id and len(n.args) == 1 and num(n.args[0]) is not None:
            return int(num(n.args[0]))
    for n in ast.walk(fn_node):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and len(n.test.ops) == 1 \
                and any(isinstance(x, ast.Raise) for b in n.body for x in ast.walk(b)):
            c = num(n.test.comparators[0])
            if c is not None and isinstance(n.test.ops[0], ast.Lt):
                return int(c) - 1
            if c is not None and isinstance(n.test.ops[0], ast.LtE):
                return int(c)
    if re.search(r"raises\(", src):
        return None
    return "unknown"


# ---------------------------------------------------------------- per test extraction
class Ctx:
    def __init__(self, mod, cls, fn):
        self.mod, self.cls, self.fn = mod, cls, fn


def call_name(mod, call):
    return mod.resolve(dotted(call.func))


def walk_body(fn):
    """ast.walk over the function including decorators."""
    yield from ast.walk(fn)


def loop_multiplier(fn):
    """Map id(node) -> literal loop multiplier for nodes inside `for _ in range(N)` loops."""
    mult = {}
    def rec(node, m):
        for ch in ast.iter_child_nodes(node):
            mm = m
            if isinstance(ch, (ast.For, ast.AsyncFor)) and isinstance(ch.iter, ast.Call) \
                    and dotted(ch.iter.func) == "range" and ch.iter.args:
                v = num(ch.iter.args[-1] if len(ch.iter.args) == 1 else ast.BinOp(ch.iter.args[1], ast.Sub(), ch.iter.args[0]))
                mm = m * (v if v is not None else 1)
            mult[id(ch)] = mm
            rec(ch, mm)
    rec(fn, 1)
    return mult


def random_assert_kinds(fn, rand_calls):
    """Classify assertions that touch random values.

    A name is 'raw' when it holds a random draw itself and 'derived' when the draw went through a
    property function first (len, set, sorted, Counter, sum, ...). An assertion is exact when it
    compares a raw random value with == / is against something that does not itself come from the
    same draw (a literal or an independent expression). Everything else that touches random values
    (membership, ranges, len/set/sorted equality, comparisons against the same draw) is property based.
    """
    rand_ids = {id(c) for c in rand_calls}
    raw, derived = set(), set()

    def kind(node):
        """'raw', 'derived' or None for an expression."""
        best = None
        def rec(x, wrapped):
            nonlocal best
            if isinstance(x, ast.Call) and (dotted(x.func) or "").split(".")[-1] in PROPERTY_WRAP and id(x) not in rand_ids:
                wrapped = True
            hit = id(x) in rand_ids or (isinstance(x, ast.Name) and x.id in raw)
            if hit and not wrapped:
                best = "raw"
            elif (hit or (isinstance(x, ast.Name) and x.id in derived)) and best is None:
                best = "derived"
            for ch in ast.iter_child_nodes(x):
                rec(ch, wrapped)
        rec(node, False)
        return best

    def mark(target, k):
        for x in ast.walk(target):
            if isinstance(x, ast.Name):
                raw.discard(x.id); derived.discard(x.id)
                if k == "raw":
                    raw.add(x.id)
                elif k == "derived":
                    derived.add(x.id)

    for n in sorted((n for n in ast.walk(fn) if hasattr(n, "lineno")), key=lambda n: (n.lineno, n.col_offset)):
        if isinstance(n, ast.Assign):
            k = kind(n.value)
            for t in n.targets:
                mark(t, k)
        elif isinstance(n, ast.AugAssign):
            k = kind(n.value)
            if k:
                mark(n.target, "derived" if isinstance(n.op, ast.BitOr) else k)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            k = kind(n.iter)
            if k:
                mark(n.target, k)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("append", "add", "extend", "update") \
                and isinstance(n.func.value, ast.Name):
            k = kind(ast.Tuple(elts=list(n.args), ctx=ast.Load()))
            if k:
                cur = "raw" if n.func.value.id in raw else None
                mark(n.func.value, "derived" if n.func.attr in ("add", "update") else (cur or k))
    out = {}
    for a in asserts_of(fn):
        for left, op, right in comparisons(a):
            if right is None:
                if kind(left):
                    out["random_assert_property"] = 1
                continue
            kl, kr = kind(left), kind(right)
            if not (kl or kr):
                continue
            if op in ("Eq", "NotEq", "Is", "IsNot") and ((kl == "raw" and not kr) or (kr == "raw" and not kl)):
                out["random_assert_exact"] = 1
            else:
                out["random_assert_property"] = 1
    return out


def body_features(fn, mod, cls_node=None):
    """Features of one function body (test or fixture)."""
    f = defaultdict(int)
    src = ast.get_source_segment(mod.src, fn) or ""
    deco_src = "\n".join(ast.unparse(d) for d in fn.decorator_list)
    cls_deco = "\n".join(ast.unparse(d) for d in (cls_node.decorator_list if cls_node else []))
    mult = loop_multiplier(fn)
    names_loaded = set()
    calls = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            names_loaded.add(n.id)
        if isinstance(n, ast.Call):
            calls.append((n, call_name(mod, n) or ""))
    lib = mod.lib

    # ---- isolation helpers
    for _n, fq in calls:
        last = fq.split(".")[-1]
        if fq.startswith("tempfile."):
            f["iso_tempfile"] = 1
        if fq.startswith(("unittest.mock.patch", "mock.patch")) or fq in ("unittest.mock.patch", "mock.patch") \
                or re.match(r"^(mocker\.patch|patch(\.object|\.dict)?)$", fq):
            f["iso_mock_patch"] = 1
        if fq.startswith("monkeypatch."):
            f["iso_monkeypatch"] = 1
        if fq.startswith("freezegun.") or last == "freeze_time" or fq.startswith("freezer."):
            f["iso_freezegun"] = 1
        if fq.startswith("time_machine.") or fq.startswith("time_machine"):
            f["iso_time_machine"] = 1
        if mod.is_lib(fq) and last[:1].isupper() and not last.endswith(("Error", "Exception", "Warning")) \
                and last not in ("Query",):
            f["fresh_lib_object"] += 1
    for n, _fq in calls:
        for k in n.keywords:
            if k.arg == "sleep" and lib == "tenacity" and (mod.resolve(dotted(k.value)) or "") not in SLEEP \
                    and not sleeps_for_real(k.value, fn, mod):
                f["iso_injected_sleep"] = 1
    if re.search(r"freeze_time", deco_src + cls_deco):
        f["iso_freezegun"] = 1
    if re.search(r"(mock\.)?patch(\.object|\.dict)?\(", deco_src + cls_deco):
        f["iso_mock_patch"] = 1
    if re.search(r"time_machine", deco_src + cls_deco):
        f["iso_time_machine"] = 1

    # ---- clock, sleep
    mocked_sleep = bool(re.search(r"(patch|setattr)\([^)]*sleep", src + deco_src)) or \
        bool(re.search(r"sleep\s*=\s*(?!time\.sleep|tenacity\.sleep|asyncio\.sleep)", src))
    clock_frozen = bool(re.search(r"freeze_time|time_machine|mock_?datetime|fake_?(clock|datetime)|(patch|setattr)\([^)]*(time|datetime|now|monotonic)",
                                  src + deco_src + cls_deco, re.I))
    f["clock_frozen_or_patched"] = int(clock_frozen)
    sleeps = []
    for n, fq in calls:
        last = fq.split(".")[-1]
        if fq in CLOCK or (isinstance(n.func, ast.Name) and n.func.id in CLOCK_HELPERS) \
                or (last in ("now", "utcnow", "today") and isinstance(n.func, ast.Attribute)) \
                or (fq == "arrow.get" and not n.args and not n.keywords):
            f["clock_reads"] += 1
        if fq in SLEEP or (last == "sleep" and fq.split(".")[0] in ("time", "asyncio", "tenacity", "anyio", "trio")):
            a = num(n.args[0]) if n.args else None
            if a is None and n.args and isinstance(n.args[0], ast.Name):
                a = mod.mod_numbers.get(n.args[0].id)
            if a is None or a > 0:
                f["sleep_calls"] += 1
                if a is not None:
                    sleeps.append(a * mult.get(id(n), 1))
    f["sleep_seconds_literal"] = round(sum(sleeps), 4)
    f["sleep_mocked"] = int(mocked_sleep)

    # ---- time of day strings built from a real clock
    for n, _fq in calls:
        if isinstance(n.func, ast.Attribute) and n.func.attr == "strftime" and n.args \
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            inner = ast.unparse(n.func.value)
            fmt = n.args[0].value
            if re.search(r"\.(now|utcnow|today)\(|time\.time\(", inner) or re.search(r"\b(now|current)\b", inner):
                f["clock_derived_string"] = 1
                if "%H" in fmt and not re.search(r"%[dYmjyb]", fmt):
                    f["clock_time_of_day_string"] = 1

    # ---- randomness
    rand_calls = []
    for n, fq in calls:
        last = fq.split(".")[-1]
        if fq in RANDOM_STD or (last in RANDOM_LIB and (mod.is_lib(fq) or fq == last)) \
                or (fq == "random.Random" and not n.args):
            rand_calls.append(n)
    f["random_calls"] = len(rand_calls)
    seeded = bool(re.search(r"\bseed\(|Random\(\s*[^)\s]|random_state\s*=|choice\s*=|(patch|setattr)\([^)]*(random|choice|uniform)",
                            src + deco_src, re.I))
    f["random_seeded_local"] = int(seeded)
    if rand_calls:
        kinds = random_assert_kinds(fn, rand_calls)
        f.update(kinds)

    # ---- concurrency
    for n, fq in calls:
        last = fq.split(".")[-1]
        if fq.startswith("threading.") and last in ("Thread", "Timer") or last in ("ThreadPoolExecutor",):
            f["threads"] = 1
        if fq.startswith("multiprocessing") or last == "ProcessPoolExecutor":
            f["processes"] = 1
        if fq.startswith("asyncio.") and last in ("gather", "create_task", "ensure_future", "wait_for", "run",
                                                  "wait", "get_event_loop", "new_event_loop", "as_completed",
                                                  "run_until_complete", "TaskGroup"):
            f["asyncio_calls"] = 1
        if fq.startswith("subprocess.") or last == "Popen":
            f["subprocess"] = 1
        if isinstance(n.func, ast.Attribute) and n.func.attr == "join" and not isinstance(n.func.value, (ast.Constant, ast.JoinedStr)) \
                and not (dotted(n.func.value) or "").startswith(("os.path", "posixpath", "str", "sep")):
            f["_join"] = 1
            if n.args or n.keywords:
                f["_join_timeout"] = 1
        if isinstance(n.func, ast.Attribute) and n.func.attr == "result" and (n.args or n.keywords):
            f["_join_timeout"] = 1
        if fq == "asyncio.wait_for":
            f["_join_timeout"] = 1
    for n in ast.walk(fn):
        if isinstance(n, (ast.With, ast.AsyncWith)):
            for it in n.items:
                if isinstance(it.context_expr, ast.Call) and (call_name(mod, it.context_expr) or "").split(".")[-1] \
                        in ("ThreadPoolExecutor", "ProcessPoolExecutor"):
                    f["_join"] = 1
    if isinstance(fn, ast.AsyncFunctionDef) or "asyncio" in deco_src:
        f["asyncio_test"] = 1

    # ---- shared state
    ylines = [n.lineno for n in ast.walk(fn) if isinstance(n, (ast.Yield, ast.YieldFrom))]
    yield_line = min(ylines) if ylines else (fn.lineno if fn.name in TEARDOWN_NAMES else 10 ** 9)
    for n in ast.walk(fn):
        if isinstance(n, ast.Global):
            f["global_stmt"] = 1
    shared_kinds = {"lib_instance", "mutable_literal", "lib_decorated_func"}
    local_names = {a.arg for a in fn.args.args} | {x.id for x in ast.walk(fn) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
    f["_mod_names"] = sorted(n for n in names_loaded - local_names if mod.mod_state.get(n) in shared_kinds)
    for name in f["_mod_names"]:
        if mod.mod_state.get(name) in ("lib_instance", "lib_decorated_func"):
            f["module_lib_object"] = 1
        else:
            f["module_data_read"] = 1
    for n in ast.walk(fn):
        tgt = []
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgt = n.targets if isinstance(n, ast.Assign) else [n.target]
        elif isinstance(n, ast.Delete):
            tgt = n.targets
        for t in tgt:
            if isinstance(t, (ast.Attribute, ast.Subscript)):
                root = t
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name):
                    if mod.mod_state.get(root.id) in shared_kinds and root.id not in local_names:
                        f["module_object_write"] = 1
                    fq = mod.resolve(dotted(t.value) if isinstance(t, ast.Attribute) else dotted(t.value)) or ""
                    if root.id in mod.alias and not fq.startswith(("os.environ", "sys.")):
                        if mod.is_lib(mod.alias[root.id]) or root.id == lib:
                            f["lib_global_write"] = 1
                        else:
                            f["other_module_attr_write"] = 1
                    if fq.startswith("os.environ"):
                        f["env_mutation"] = 1
                    if fq.startswith("sys") or (isinstance(t, ast.Attribute) and dotted(t.value) == "sys"):
                        f["env_mutation"] = 1
                    if cls_node is not None and root.id in (cls_node.name, "cls"):
                        f["class_attr_write"] = 1
                    if dotted(t) and re.match(r"^(self\.__class__|type\(self\)\(\))\.", dotted(t) or ""):
                        f["class_attr_write"] = 1
                    if (dotted(t.value) or "").startswith("request.cls"):
                        f["class_attr_write"] = 1
    for n, fq in calls:
        last = fq.split(".")[-1]
        if fq in ("os.putenv", "os.unsetenv", "os.chdir", "locale.setlocale", "time.tzset", "sys.setrecursionlimit",
                  "socket.setdefaulttimeout") or re.match(r"^os\.environ\.(update|pop|setdefault|clear|__setitem__|popitem)$", fq) \
                or re.match(r"^sys\.(path|modules|meta_path)\.(insert|append|extend|remove|pop|update|__setitem__)$", fq):
            f["env_mutation"] = 1
        if fq == "setattr" and n.args:
            r = mod.resolve(dotted(n.args[0])) or ""
            if mod.is_lib(r):
                f["lib_global_write"] = 1
        if isinstance(n.func, ast.Attribute) and n.func.attr in MUTATING:
            root = n.func.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name) and mod.mod_state.get(root.id) in shared_kinds and root.id not in local_names:
                f["module_object_write"] = 1
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and mod.mod_state.get(n.func.value.id) in ("lib_instance", "lib_decorated_func") \
                and n.func.value.id not in local_names:
            f["module_object_write"] = 1
        if isinstance(n.func, ast.Name) and mod.mod_state.get(n.func.id) == "lib_decorated_func" and n.func.id not in local_names:
            f["module_object_write"] = 1   # calling a module level @cached / @retry / breaker function updates its state
        # library singletons and registries
        is_reset = (lib == "schedule" and fq == "schedule.clear") or \
            (lib == "humanize" and last == "deactivate" and mod.is_lib(fq))
        if is_reset:
            after = n.lineno > yield_line or in_finally(fn, n)
            f["_reset_after" if after else "_reset_before"] = 1
        if lib == "schedule" and fq.startswith("schedule.") and last in SCHEDULE_DEFAULT and last != "clear":
            f["lib_singleton_use"] = 1
        if lib == "filelock":
            for k in n.keywords:
                if k.arg == "is_singleton" and isinstance(k.value, ast.Constant) and k.value.value is True:
                    f["lib_singleton_use"] = 1
        if lib == "humanize" and last == "activate" and mod.is_lib(fq):
            f["lib_singleton_use"] = 1
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute):
            fq = mod.resolve(dotted(n)) or ""
            if lib == "schedule" and fq.startswith("schedule.") and n.attr in ("jobs", "default_scheduler"):
                f["lib_singleton_use"] = 1
            if lib == "arrow" and n.attr == "_locale_map":
                f["lib_singleton_use"] = 1
            if lib == "filelock" and n.attr == "_instances":
                f["lib_singleton_use"] = 1
            if lib in LIB_FLAGS and n.attr in LIB_FLAGS[lib] and isinstance(n.ctx, (ast.Store, ast.Del)):
                f["lib_global_write"] = 1
        if isinstance(n, ast.ClassDef) and lib == "arrow":
            for b in n.bases:
                if (dotted(b) or "").endswith("Locale"):
                    f["lib_singleton_use"] = 1
                    f["lib_global_write"] = 1   # Locale subclasses register themselves in _locale_map
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute) and n.attr.startswith("_") and not n.attr.startswith("__") \
                and not (isinstance(n.value, ast.Name) and n.value.id in ("self", "cls")):
            f["private_attr_access"] = 1
    if cls_node is not None:
        cattrs = {}
        for st in cls_node.body:
            if isinstance(st, ast.Assign):
                for t in st.targets:
                    if isinstance(t, ast.Name):
                        cattrs[t.id] = mod._kind(st.value)
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" \
                    and cattrs.get(n.attr) in shared_kinds:
                f["class_attr_read"] = 1
    return f


TEARDOWN_NAMES = {"tearDown", "teardown_method", "teardown", "tearDownClass", "teardown_class", "asyncTearDown"}


def in_finally(fn, node):
    for t in ast.walk(fn):
        if isinstance(t, ast.Try):
            for st in t.finalbody:
                if any(x is node for x in ast.walk(st)):
                    return True
    return False


def asserts_of(fn):
    for n in ast.walk(fn):
        if isinstance(n, ast.Assert):
            yield n.test
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("assert") \
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "self":
            yield n


UNITTEST_OPS = {"assertLess": "Lt", "assertLessEqual": "LtE", "assertGreater": "Gt", "assertGreaterEqual": "GtE",
                "assertEqual": "Eq", "assertNotEqual": "NotEq", "assertIs": "Is", "assertIn": "In",
                "assertAlmostEqual": "Approx"}


def comparisons(a):
    """Yield (left, op, right) for every comparison in an assert expression."""
    if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr.startswith("assert") \
            and isinstance(a.func.value, ast.Name) and a.func.value.id == "self":
        op = UNITTEST_OPS.get(a.func.attr)
        if op and len(a.args) >= 2:
            yield a.args[0], op, a.args[1]
        elif a.func.attr in ("assertTrue", "assertFalse") and a.args:
            yield from comparisons(a.args[0])
        return
    for n in ast.walk(a):
        if isinstance(n, ast.Compare):
            left = n.left
            for op, right in zip(n.ops, n.comparators, strict=True):
                yield left, type(op).__name__, right
                left = right
        elif isinstance(n, ast.Call) and (dotted(n.func) or "").split(".")[-1] in ("all", "any", "isinstance"):
            yield n, "Call", None


def count_asserts(fn):
    n_assert = n_raises = 0
    for n in ast.walk(fn):
        if isinstance(n, ast.Assert):
            n_assert += 1
        elif isinstance(n, ast.Call):
            d = dotted(n.func) or ""
            if d.startswith("self.assert") or d.startswith("self.fail"):
                if "Raises" in d:
                    n_raises += 1
                else:
                    n_assert += 1
            elif d.endswith("raises") or d.endswith("warns"):
                n_raises += 1
    return n_assert, n_raises


def parametrize_cases(fn):
    total, any_p = 1, False
    for d in fn.decorator_list:
        if isinstance(d, ast.Call) and (dotted(d.func) or "").endswith("parametrize"):
            any_p = True
            if len(d.args) >= 2 and isinstance(d.args[1], (ast.List, ast.Tuple)):
                total *= max(1, len(d.args[1].elts))
    return any_p, (total if any_p else 1)


# ---------------------------------------------------------------- timing assertions
def timing_asserts(fn, mod, cond, proj, test):
    clock_vars, elapsed_vars = {}, {}
    def is_clock(node):
        return isinstance(node, ast.Call) and (call_name(mod, node) in ELAPSED_CLOCK or
                                               call_name(mod, node) in CLOCK or
                                               (isinstance(node.func, ast.Name) and node.func.id in CLOCK_HELPERS))

    def scale_of(node):
        for x in ast.walk(node):
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in CLOCK_HELPERS:
                return CLOCK_HELPERS[x.func.id]
        return 1.0
    def mentions_clock(node):
        return any(is_clock(x) or (isinstance(x, ast.Name) and x.id in clock_vars) for x in ast.walk(node))
    stmts = sorted((n for n in ast.walk(fn) if isinstance(n, ast.stmt)), key=lambda s: s.lineno)
    for s in stmts:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            v, name = s.value, s.targets[0].id
            if is_clock(v):
                clock_vars.setdefault(name, []).append(s.lineno)
            elif isinstance(v, ast.BinOp) and isinstance(v.op, ast.Sub) and mentions_clock(v):
                starts = [x.id for x in ast.walk(v.right) if isinstance(x, ast.Name) and x.id in clock_vars]
                elapsed_vars[name] = (s.lineno, starts[0] if starts else None, v)
            elif isinstance(v, ast.Call) and (dotted(v.func) or "").endswith("total_seconds") and isinstance(v.func, ast.Attribute) \
                    and isinstance(v.func.value, ast.BinOp) and mentions_clock(v.func.value):
                starts = [x.id for x in ast.walk(v.func.value.right) if isinstance(x, ast.Name) and x.id in clock_vars]
                elapsed_vars[name] = (s.lineno, starts[0] if starts else None, v)
    env = dict(mod.mod_numbers)
    for s in stmts:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            v = num(s.value, env)
            if v is not None:
                env[s.targets[0].id] = v

    def elapsed_side(node, line):
        """Return (start var, end line) if node is a measured elapsed time."""
        if isinstance(node, ast.Name) and node.id in elapsed_vars:
            ln, start, _ = elapsed_vars[node.id]
            return start, ln
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) and mentions_clock(node):
            starts = [x.id for x in ast.walk(node.right) if isinstance(x, ast.Name) and x.id in clock_vars]
            return (starts[0] if starts else None), line
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and node.slice.value in ("delay_since_first_attempt",):
            return "__retry_start__", line
        return None

    out = []
    for a in asserts_of(fn):
        line = a.lineno
        for left, op, right in comparisons(a):
            if right is None:
                continue
            tol = None
            for side, other, flip in ((left, right, False), (right, left, True)):
                es = elapsed_side(side, line)
                if es is None:
                    continue
                start, end_line = es
                if start == "__retry_start__" and op not in ("Lt", "LtE", "Gt", "GtE"):
                    continue
                thr = num(other, env)
                sc = scale_of(side) if not isinstance(side, ast.Name) else \
                    scale_of(elapsed_vars[side.id][2]) if side.id in elapsed_vars else 1.0
                if op == "Approx" or (isinstance(other, ast.Call) and (dotted(other.func) or "").endswith("approx")):
                    if isinstance(other, ast.Call) and (dotted(other.func) or "").endswith("approx"):
                        thr = num(other.args[0], env) if other.args else None
                        tol = kw(other, "abs", env=env)
                        r = kw(other, "rel", env=env)
                        if tol is None and thr is not None:
                            tol = (r if r is not None else 1e-6) * thr
                    else:
                        tol = kw(a, "delta", env=env)
                    direction = "approx"
                elif op in ("Lt", "LtE"):
                    direction = "lower" if flip else "upper"
                elif op in ("Gt", "GtE"):
                    direction = "upper" if flip else "lower"
                elif op == "Eq":
                    direction = "approx"
                else:
                    continue
                # window: last assignment of the start var before the measurement
                if start == "__retry_start__":
                    w0 = fn.lineno
                elif start and start in clock_vars:
                    prior = [l for l in clock_vars[start] if l <= end_line]
                    w0 = prior[-1] if prior else fn.lineno
                else:
                    w0 = fn.lineno
                if sc != 1.0:
                    thr = None if thr is None else thr / sc
                    tol = None if tol is None else tol / sc
                rec = {"line": line, "direction": direction, "threshold": thr, "tolerance": tol, "unit_scale": sc,
                       "expr": ast.unparse(a)[:160], "window": [w0, end_line]}
                man = MANUAL_DELAY.get((cond, proj, test, line, direction))
                if man:
                    if man[0] is not None:
                        rec["threshold"] = man[0]
                    rec.update({"planned": list(man[1:4]), "delay_method": "manual: " + man[4]})
                else:
                    rec.update(planned_delay(fn, mod, w0, end_line, env))
                rec["slack_ms"] = slack(rec)
                out.append(rec)
                break
    return out


def slack(rec):
    thr, d = rec["threshold"], rec.get("planned")
    if thr is None or d is None:
        return None
    lo, _, hi = d
    if rec["direction"] == "upper":
        return round(1000 * (thr - hi), 1)
    if rec["direction"] == "lower":
        return round(1000 * (lo - thr), 1)
    tol = rec.get("tolerance") or 0
    return round(1000 * min(hi and (thr + tol - hi), lo - (thr - tol)), 1)


def sleeps_for_real(node, fn, mod):
    """Does an injected tenacity sleep= callable really sleep? Follows a local name to its constructor
    and a module level class or function to its body."""
    name = None
    if isinstance(node, ast.Name):
        name = node.id
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets) \
                    and isinstance(n.value, ast.Call):
                name = dotted(n.value.func) or name
    elif isinstance(node, ast.Call):
        name = dotted(node.func)
    elif isinstance(node, ast.Lambda):
        return bool(re.search(r"(time|asyncio)\.sleep\(", ast.unparse(node)))
    for top in mod.tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and top.name == name:
            return bool(re.search(r"(time|asyncio)\.sleep\(|\bsleep\(", ast.unparse(top)))
    return False


def planned_delay(fn, mod, w0, w1, env):
    mult = loop_multiplier(fn)
    in_win = lambda n: hasattr(n, "lineno") and w0 <= n.lineno <= w1
    sleeps, sources = [], []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and in_win(n):
            fq = call_name(mod, n) or ""
            last = fq.split(".")[-1]
            if (fq in SLEEP or (last == "sleep" and fq.split(".")[0] in ("time", "asyncio", "tenacity"))) and n.args:
                v = num(n.args[0], env)
                if v is None:
                    return {"planned": None, "delay_method": "sleep with non literal argument"}
                sleeps.append(v * mult.get(id(n), 1))
            if isinstance(n.func, ast.Attribute) and n.func.attr in ("wait", "join") and (n.args or n.keywords) \
                    and not isinstance(n.func.value, ast.Constant):
                sources.append("blocking wait/join")
    fn_src = ast.unparse(fn)
    # tenacity configs anywhere in the test
    configs = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            last = (call_name(mod, n) or "").split(".")[-1]
            if last in ("retry", "Retrying", "AsyncRetrying", "async_retry") and (mod.lib == "tenacity"):
                if any(k.arg in ("wait", "stop") for k in n.keywords):
                    configs.append(n)
    if configs and mod.lib == "tenacity":
        keyset = {ast.unparse(c) for c in configs}
        if len(keyset) > 1:
            return {"planned": None, "delay_method": "several retry configs in one test"}
        c = configs[0]
        wn = next((k.value for k in c.keywords if k.arg == "wait"), None)
        sn = next((k.value for k in c.keywords if k.arg == "stop"), None)
        waits, stop = tenacity_waits(wn, mod, env), tenacity_stop(sn, mod, env)
        sl = next((k.value for k in c.keywords if k.arg == "sleep"), None)
        if sl is not None and (mod.resolve(dotted(sl)) or "") not in SLEEP and not sleeps_for_real(sl, fn, mod):
            waits = lambda a: (0.0, 0.0, 0.0)   # injected fake sleep
        fails = infer_failures(fn, mod)
        if waits is None or stop is None or fails == "unknown":
            return {"planned": None, "delay_method": "tenacity config not modelled"}
        sim = simulate(waits, stop, fails)
        if sim is None:
            return {"planned": None, "delay_method": "tenacity never stops"}
        tot = [round(x + sum(sleeps), 4) for x in sim]
        return {"planned": tot, "delay_method": f"tenacity simulation ({'always fails' if fails is None else f'{fails} failures'})"
                + (" + sleeps" if sleeps else "")}
    if mod.lib == "filelock" and re.search(r"raises\(\s*(filelock\.)?Timeout", fn_src):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and in_win(n):
                t = kw(n, "timeout", env=env)
                if t is not None and t > 0:
                    return {"planned": [t, t, t], "delay_method": "filelock acquire timeout expected to expire"}
    if mod.lib == "more_itertools":
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and (call_name(mod, n) or "").endswith("time_limited"):
                t = kw(n, "limit_seconds", 0, env=env)
                if t is not None:
                    if re.search(r"\b(count|cycle|repeat)\(|while True", fn_src) or sleeps:
                        extra = max(sleeps) if sleeps else 0.0
                        return {"planned": [min(t, sum(sleeps)) if sleeps else t, t, t + extra],
                                "delay_method": "time_limited runs until its limit (+ one in-flight sleep)"}
                    return {"planned": [0.0, 0.0, 0.0], "delay_method": "time_limited over a short finite iterable"}
    if mod.lib == "schedule":
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and in_win(n) and isinstance(n.func, ast.Attribute) and n.func.attr == "run_all":
                dsec = kw(n, "delay_seconds", 0, 0.0, env)
                jobs = sum(1 for x in ast.walk(fn) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                           and x.func.attr == "do" and x.lineno < n.lineno)
                d = dsec * jobs
                return {"planned": [d, d, d], "delay_method": f"run_all sleeps delay_seconds after each of {jobs} jobs"}
    if sources:
        return {"planned": None, "delay_method": "blocking wait or join in window"}
    tot = round(sum(sleeps), 4)
    return {"planned": [tot, tot, tot], "delay_method": "literal sleeps in window" if sleeps else "no delay found in window"}


# ---------------------------------------------------------------- suite walk
CLOCK_HELPERS = {}   # helper function name -> unit scale (1 = seconds, 1000 = ms), per suite


def scan_clock_helpers(paths):
    """Module level helpers in the suite's .py files that just return a clock reading."""
    out = {}
    for d in {os.path.dirname(p) for p in paths}:
        for f in glob.glob(os.path.join(d, "*.py")):
            try:
                t = ast.parse(open(f, encoding="utf8", errors="replace").read())
            except SyntaxError:
                continue
            for node in t.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("test"):
                    rets = [ast.unparse(r.value) for r in ast.walk(node) if isinstance(r, ast.Return) and r.value is not None]
                    if len(rets) == 1 and re.search(r"time\.(time|monotonic|perf_counter)(_ns)?\(", rets[0]) and len(node.body) <= 3:
                        scale = 1000.0 if re.search(r"\*\s*1000\b", rets[0]) else 1e9 if "_ns(" in rets[0] else 1.0
                        out[node.name] = scale
    return out


def conftests(path, cond, proj):
    stop = f"{R}/subjects" if cond == "human" else f"{R}/{cond}"
    out, d = [], os.path.dirname(path)
    while d.startswith(stop) and len(d) > len(stop):
        c = os.path.join(d, "conftest.py")
        if os.path.exists(c):
            out.append(c)
        d = os.path.dirname(d)
    return out


def fixture_closure(names, mod, seen=None):
    seen = seen if seen is not None else {}
    for nm in names:
        if nm in seen:
            continue
        if nm in mod.fixtures:
            node, fmod = mod.fixtures[nm]
            info = fixture_info(node, fmod)
            seen[nm] = (node, fmod, info)
            fixture_closure(info["args"], fmod if fmod is not mod else mod, seen)
        else:
            seen[nm] = None
    return seen


def usefixtures(decos):
    out = []
    for d in decos:
        if isinstance(d, ast.Call) and (dotted(d.func) or "").endswith("usefixtures"):
            out += [a.value for a in d.args if isinstance(a, ast.Constant)]
    return out


def suite_tests(cond, proj):
    recs, coll_err = [], []
    CLOCK_HELPERS.clear(); CLOCK_HELPERS.update(scan_clock_helpers(suite_files(cond, proj)))
    for path in suite_files(cond, proj):
        try:
            mod = Module(path, proj, conftests(path, cond, proj))
        except SyntaxError:
            coll_err.append(path); continue
        autouse_mod = [k for k, (node, fm) in mod.fixtures.items() if fixture_info(node, fm)["autouse"]]
        def visit(node, cls_stack, path=path, mod=mod, autouse_mod=autouse_mod):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.ClassDef):
                    visit(ch, cls_stack + [ch])
                elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) and ch.name.startswith("test"):
                    recs.append(test_record(cond, proj, path, mod, ch, cls_stack, autouse_mod))
        visit(mod.tree, [])
    return recs, coll_err


def test_record(cond, proj, path, mod, fn, cls_stack, autouse_mod):
    cls = cls_stack[-1] if cls_stack else None
    f = body_features(fn, mod, cls)
    # class level: setup/teardown methods, class fixtures, unittest style
    setup_fns, teardown_fns, cls_fixture_names, cls_autouse, teardown_nodes = [], set(), [], [], []
    for c in cls_stack:
        for st in c.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if st.name in ("setUp", "setup_method", "setup", "setUpClass", "setup_class", "asyncSetUp"):
                    setup_fns.append(st)
                if st.name in TEARDOWN_NAMES:
                    teardown_fns.add(st.name); teardown_nodes.append(st)
                info = fixture_info(st, mod)
                if info and info["autouse"]:
                    cls_autouse.append(st)
        cls_fixture_names += usefixtures(c.decorator_list)
    pairs = {"setUp": ("tearDown",), "setup_method": ("teardown_method",), "setup": ("teardown",),
             "setUpClass": ("tearDownClass",), "setup_class": ("teardown_class",), "asyncSetUp": ("asyncTearDown",)}
    unmatched = [s for s in setup_fns if not any(t in teardown_fns for t in pairs[s.name])]
    if any(m in mod.mod_setup for m in ("setup_module", "setup_function", "setUpModule")) and not \
            any(m in mod.mod_setup for m in ("teardown_module", "teardown_function", "tearDownModule")):
        f["setup_without_teardown"] = 1
    f["setup_without_teardown"] = int(bool(unmatched) or f.get("setup_without_teardown", 0))
    # merge features from setup methods and fixtures
    merged = [body_features(s, mod, cls) for s in setup_fns + cls_autouse + teardown_nodes]
    req = [a.arg for a in fn.args.args if a.arg not in ("self", "cls")] + usefixtures(fn.decorator_list) \
        + cls_fixture_names + autouse_mod
    clos = fixture_closure(req, mod)
    builtin_used = set()
    fx_meta = []
    via = {}
    for nm, v in clos.items():
        if v is None:
            if nm in ISO_FIXTURES:
                builtin_used.add(ISO_FIXTURES[nm])
            continue
        node, fmod, info = v
        ff = body_features(node, fmod)
        ff.pop("class_attr_write", None)   # request.cls.x = ... in a fixture: counted as wide_scope_fixture
        if info["scope"] != "function":
            ff["wide_scope_fixture"] = 1
            ff["fresh_lib_object"] = 0   # an object built once per module/class is shared, not fresh
        ff["_yield"] = int(info["yields"])
        merged.append(ff)
        fx_meta.append(nm)
    MERGE = ["iso_injected_sleep", "iso_tempfile", "iso_mock_patch", "iso_monkeypatch", "iso_freezegun", "iso_time_machine",
             "fresh_lib_object", "clock_frozen_or_patched", "lib_singleton_use", "lib_global_write", "env_mutation",
             "module_lib_object", "module_data_read", "module_object_write", "class_attr_write", "wide_scope_fixture",
             "_reset_before", "_reset_after", "sleep_mocked", "random_seeded_local", "private_attr_access"]
    # helpers the test calls (self.<method> in any class of the module, or module level functions):
    # only concurrency and sleep flags are taken from them
    helper_names = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"}
    helper_names |= {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    helpers = [h for h in ast.walk(mod.tree) if isinstance(h, (ast.FunctionDef, ast.AsyncFunctionDef))
               and h.name in helper_names and not h.name.startswith("test") and h is not fn
               and h.name not in mod.fixtures]
    for h in helpers[:20]:
        hf = body_features(h, mod)
        for k in ("threads", "processes", "asyncio_calls", "_join", "_join_timeout", "subprocess", "sleep_calls"):
            if hf.get(k) and not f.get(k):
                f[k] = hf[k]
                via["helper:" + k] = 1
    for ff in merged:
        for k in MERGE:
            if ff.get(k):
                if not f.get(k):
                    via[k] = 1
                f[k] = max(f.get(k, 0), 1 if k != "fresh_lib_object" else ff[k])
    for b in builtin_used:
        f["iso_" + b] = 1
    has_yield_fixture = any(ff.get("_yield") for ff in merged)
    # module level seed
    mod_src = mod.src
    if re.search(r"^random\.seed\(", mod_src, re.M) or re.search(r"autouse\s*=\s*True[\s\S]{0,300}random\.seed", mod_src):
        f["random_seeded_local"] = 1
    if re.search(r"autouse\s*=\s*True[\s\S]{0,400}(freeze_time|time_machine)", mod_src):
        f["clock_frozen_or_patched"] = 1; f["iso_freezegun"] = 1
    # derived flags
    n_assert, n_raises = count_asserts(fn)
    is_p, cases = parametrize_cases(fn)
    timing = timing_asserts(fn, mod, cond, proj, fn.name)
    frozen_timing = 0
    if timing and f.get("clock_frozen_or_patched"):
        frozen_timing, timing = len(timing), []   # comparisons against a frozen clock are deterministic
    d = {
        "cond": cond, "label": CONDS[cond], "proj": proj, "file": os.path.relpath(fn and mod.path, R),
        "class": cls.name if cls else None, "test": fn.name, "line": fn.lineno,
        # size and style
        "lines": fn.end_lineno - fn.lineno + 1, "n_asserts": n_assert, "n_raises": n_raises,
        "parametrized": int(is_p), "param_cases": cases, "in_class": int(cls is not None),
        "unittest_style": int(any((dotted(b) or "").endswith("TestCase") for c in cls_stack for b in c.bases)),
        "is_async": int(isinstance(fn, ast.AsyncFunctionDef)),
        "fixtures": sorted(fx_meta), "builtin_fixtures": sorted(builtin_used),
        # shared state
        "global_stmt": f.get("global_stmt", 0), "module_lib_object": f.get("module_lib_object", 0),
        "module_data_read": f.get("module_data_read", 0), "module_names_used": f.get("_mod_names", []),
        "lib_global_reset": int(bool(f.get("_reset_before") or f.get("_reset_after"))),
        "lib_global_reset_after": f.get("_reset_after", 0),
        "module_object_write": f.get("module_object_write", 0), "class_attr_read": f.get("class_attr_read", 0),
        "class_attr_write": f.get("class_attr_write", 0), "lib_global_write": f.get("lib_global_write", 0),
        "lib_singleton_use": f.get("lib_singleton_use", 0), "env_mutation_unguarded": f.get("env_mutation", 0),
        "setup_without_teardown": f.get("setup_without_teardown", 0), "wide_scope_fixture": f.get("wide_scope_fixture", 0),
        "private_attr_access": f.get("private_attr_access", 0),
        # isolation
        "iso_tmp_path": f.get("iso_tmp_path", 0), "iso_tempfile": f.get("iso_tempfile", 0),
        "iso_monkeypatch": f.get("iso_monkeypatch", 0), "iso_freezegun": f.get("iso_freezegun", 0),
        "iso_time_machine": f.get("iso_time_machine", 0), "iso_mock_patch": f.get("iso_mock_patch", 0),
        "iso_injected_sleep": f.get("iso_injected_sleep", 0),
        "fresh_lib_objects": f.get("fresh_lib_object", 0), "yield_fixture": int(has_yield_fixture),
        # time
        "clock_reads": f.get("clock_reads", 0), "clock_frozen_or_patched": f.get("clock_frozen_or_patched", 0),
        "sleep_calls": 0 if f.get("sleep_mocked") else f.get("sleep_calls", 0),
        "sleep_seconds_literal": 0 if f.get("sleep_mocked") else f.get("sleep_seconds_literal", 0),
        "clock_derived_string": f.get("clock_derived_string", 0),
        "clock_time_of_day_string": f.get("clock_time_of_day_string", 0) and not f.get("clock_frozen_or_patched"),
        "timing_asserts": timing, "timing_asserts_on_frozen_clock": frozen_timing,
        # randomness
        "random_calls": f.get("random_calls", 0), "random_seeded": f.get("random_seeded_local", 0),
        "random_assert_exact": f.get("random_assert_exact", 0), "random_assert_property": f.get("random_assert_property", 0),
        # concurrency
        "threads": f.get("threads", 0), "processes": f.get("processes", 0),
        "asyncio": int(bool(f.get("asyncio_calls") or f.get("asyncio_test"))), "subprocess": f.get("subprocess", 0),
        "joins": f.get("_join", 0), "join_timeout": f.get("_join_timeout", 0),
        "via_fixture_or_setup": sorted(via),
    }
    d["clock_time_of_day_string"] = int(bool(d["clock_time_of_day_string"]))
    # summary flags
    d["real_clock_read"] = int(d["clock_reads"] > 0 and not d["clock_frozen_or_patched"])
    d["real_sleep"] = int(d["sleep_calls"] > 0)
    d["timing_assert"] = int(bool(timing))
    d["random_unseeded"] = int(d["random_calls"] > 0 and not d["random_seeded"])
    d["random_unseeded_exact"] = int(d["random_unseeded"] and d["random_assert_exact"])
    d["concurrency"] = int(bool(d["threads"] or d["processes"] or d["asyncio"]))
    d["threads_or_procs_without_join"] = int(bool((d["threads"] or d["processes"]) and not d["joins"]))
    d["lib_singleton_unreset"] = int(bool(d["lib_singleton_use"] and not d["lib_global_reset_after"]))
    d["global_state_write"] = int(bool(d["global_stmt"] or d["module_object_write"] or d["class_attr_write"]
                                       or d["lib_global_write"] or d["env_mutation_unguarded"]))
    d["shared_state_any"] = int(bool(d["global_state_write"] or d["module_lib_object"] or d["class_attr_read"]
                                     or d["lib_singleton_use"] or d["wide_scope_fixture"]))
    d["isolation_helper_any"] = int(bool(d["iso_tmp_path"] or d["iso_tempfile"] or d["iso_monkeypatch"] or d["iso_freezegun"]
                                         or d["iso_time_machine"] or d["iso_mock_patch"] or d["iso_injected_sleep"]))
    d["fresh_object"] = int(d["fresh_lib_objects"] > 0)
    return d


# ---------------------------------------------------------------- flaky labels and aggregation
def flaky_keys():
    s = json.load(open(f"{R}/results/summary.json"))
    keys = []
    for fl in s["flaky"]:
        left, _, name = fl["test"].partition("::")
        name = name.split("[")[0]
        parts = left.split(".")
        cls = parts[-1] if parts[-1][:1].isupper() else None
        modname = parts[-2] if cls else parts[-1]
        keys.append((fl["cond"], fl["proj"], modname, cls, name, fl["test"]))
    return keys


FEATURES = [
    ("shared_state_any", "any shared state"), ("global_state_write", "writes global state"),
    ("module_lib_object", "uses module level library object"), ("module_data_read", "reads module level list/dict data"),
    ("module_object_write", "writes module level object"),
    ("class_attr_read", "reads mutable class attribute"), ("class_attr_write", "writes class attribute"),
    ("lib_singleton_use", "library singleton or registry"),
    ("lib_singleton_unreset", "singleton/registry, no reset after"),
    ("lib_global_reset", "resets library global (clear/deactivate)"),
    ("lib_global_reset_after", "resets library global after test"),
    ("lib_global_write", "writes library global"),
    ("env_mutation_unguarded", "environ/locale/sys/cwd mutation"), ("setup_without_teardown", "setup without teardown"),
    ("wide_scope_fixture", "module/class/session fixture"),
    ("private_attr_access", "touches private attribute (x._name)"),
    ("isolation_helper_any", "any isolation helper"), ("iso_tmp_path", "tmp_path/tmpdir"), ("iso_tempfile", "tempfile"),
    ("iso_monkeypatch", "monkeypatch"), ("iso_freezegun", "freezegun"), ("iso_mock_patch", "mock.patch"),
    ("iso_injected_sleep", "fake sleep injected (tenacity sleep=)"),
    ("fresh_object", "builds fresh library object"), ("yield_fixture", "yield fixture (teardown)"),
    ("real_clock_read", "real clock read"), ("real_sleep", "real sleep"), ("timing_assert", "timing assertion"),
    ("clock_time_of_day_string", "clock derived time of day string"),
    ("random_calls", "random call"), ("random_unseeded", "unseeded random"),
    ("random_assert_exact", "exact assert on random value"), ("random_unseeded_exact", "unseeded + exact"),
    ("concurrency", "threads/processes/asyncio"), ("threads", "threads"), ("asyncio", "asyncio"),
    ("threads_or_procs_without_join", "threads without join"),
    ("parametrized", "parametrized"), ("in_class", "in a class"), ("unittest_style", "unittest.TestCase"),
]
NUMERIC = ["lines", "n_asserts", "n_raises", "param_cases"]


def aggregate(recs):
    n = len(recs)
    if not n:
        return {}
    a = {"n_tests": n}
    for k, _ in FEATURES:
        a[k] = round(100 * sum(1 for r in recs if r.get(k)) / n, 2)
        a[k + "_count"] = sum(1 for r in recs if r.get(k))
    for k in NUMERIC:
        vals = sorted(r[k] for r in recs)
        a[k + "_mean"] = round(sum(vals) / n, 2)
        a[k + "_median"] = vals[n // 2]
    a["zero_assert_tests"] = sum(1 for r in recs if r["n_asserts"] == 0 and r["n_raises"] == 0)
    ta = [t for r in recs for t in r["timing_asserts"]]
    a["timing_assertions_total"] = len(ta)
    a["timing_tests_two_sided"] = sum(1 for r in recs if {t["direction"] for t in r["timing_asserts"]} >= {"upper", "lower"})
    a["timing_upper"] = sum(1 for t in ta if t["direction"] == "upper")
    a["timing_lower"] = sum(1 for t in ta if t["direction"] == "lower")
    sl = [t["slack_ms"] for t in ta if t["slack_ms"] is not None]
    a["timing_with_slack"] = len(sl)
    up = sorted(t["slack_ms"] for t in ta if t["slack_ms"] is not None and t["direction"] == "upper")
    lo = sorted(t["slack_ms"] for t in ta if t["slack_ms"] is not None and t["direction"] == "lower")
    a["upper_slack_ms"] = up
    a["lower_slack_ms"] = lo
    a["upper_slack_min_ms"] = up[0] if up else None
    a["upper_slack_median_ms"] = up[len(up) // 2] if up else None
    a["upper_slack_lt_100ms"] = sum(1 for x in up if x < 100)
    a["lower_slack_negative"] = sum(1 for x in lo if x < 0)
    return a


def classify_time_arg(node, mod, frozen, env_assign=None):
    src = ast.unparse(node)
    for x in ast.walk(node):   # follow one level of local names (future = now() + ...)
        if isinstance(x, ast.Name) and env_assign and x.id in env_assign:
            src += " " + ast.unparse(env_assign[x.id])
    clock = bool(re.search(r"\.(now|utcnow|today)\(|time\.time\(", src))
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "string literal"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "strftime":
        if clock:
            return "string from frozen clock" if frozen else "string from real clock"
        return "string from fixed datetime"
    if "timedelta(" in src and not clock:
        return "timedelta"
    if clock:
        return "datetime from frozen clock" if frozen else "datetime from real clock"
    if re.search(r"\btime\(", src) and "datetime(" not in src:
        return "datetime.time"
    if "datetime(" in src:
        return "fixed datetime"
    return "other: " + src[:40]


def case_studies(recs):
    """Per condition patterns next to the three OS-H flaky tests."""
    out = {"schedule_until_args": {}, "schedule_at_args": {}, "tenacity_random_wait_with_timing_assert": {},
           "tenacity_stop_delay_upper_bounds": {}}
    for r in recs:
        lab = r["label"]
        if r["proj"] == "schedule":
            mod = Module(os.path.join(R, r["file"]), "schedule")
            fn = next((n for n in ast.walk(mod.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and n.name == r["test"] and n.lineno == r["line"]), None)
            if fn is None:
                continue
            env_assign = {}
            for n in ast.walk(fn):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    env_assign[n.targets[0].id] = n.value
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("until", "at") and n.args:
                    a = n.args[0]
                    if isinstance(a, ast.Name) and a.id in env_assign:
                        a = env_assign[a.id]
                    kind = classify_time_arg(a, mod, bool(r["clock_frozen_or_patched"]), env_assign)
                    key = "schedule_until_args" if n.func.attr == "until" else "schedule_at_args"
                    out[key].setdefault(lab, {}).setdefault(kind, 0)
                    out[key][lab][kind] += 1
        if r["proj"] == "tenacity" and r["timing_asserts"]:
            src = open(os.path.join(R, r["file"])).read().splitlines()[r["line"] - 1: r["line"] - 1 + r["lines"]]
            body = "\n".join(src)
            if re.search(r"wait_random|wait_exponential_jitter|wait_full_jitter", body):
                out["tenacity_random_wait_with_timing_assert"].setdefault(lab, []).append(
                    {"test": r["test"], "flaky": r["flaky"],
                     "asserts": [(t["direction"], t["threshold"], t["planned"], t["slack_ms"]) for t in r["timing_asserts"]]})
            if re.search(r"stop_(before|after)_delay", body):
                ups = [t["slack_ms"] for t in r["timing_asserts"] if t["direction"] == "upper" and t["slack_ms"] is not None]
                out["tenacity_stop_delay_upper_bounds"].setdefault(lab, []).append(
                    {"test": r["test"], "flaky": r["flaky"], "upper_slack_ms": ups,
                     "lower_slack_ms": [t["slack_ms"] for t in r["timing_asserts"] if t["direction"] == "lower"]})
    return out


def main():
    all_recs, suites = [], {}
    for cond in CONDS:
        for proj in LIBS:
            recs, coll = suite_tests(cond, proj)
            if not recs and not coll:
                continue
            all_recs += recs
            suites[f"{cond}/{proj}"] = {"n_test_functions": len(recs), "syntax_error_files": coll}
    # flaky labels
    fk = flaky_keys()
    for r in all_recs:
        r["flaky"] = 0
    matched = {}
    for cond, proj, modname, cls, name, tid in fk:
        cands = [r for r in all_recs if r["cond"] == cond and r["proj"] == proj and r["test"] == name
                 and os.path.splitext(os.path.basename(r["file"]))[0] == modname]
        exact = [r for r in cands if r["class"] == cls] or cands
        for r in exact[:1]:
            r["flaky"] = 1
            r.setdefault("flaky_ids", []).append(tid)
        matched[tid] = bool(exact)
    # filter results (one run filter, collection errors)
    for cond in ("gen_oneshot_sonnet", "gen_oneshot_haiku", "gen_tp_sonnet", "gen_plain_sonnet",
                 "gen_plain_gpt54mini", "gen_plain_gpt41mini"):
        for proj in LIBS:
            fp = f"{R}/results/filter/{cond}/{proj}.json"
            if os.path.exists(fp) and f"{cond}/{proj}" in suites:
                fj = json.load(open(fp))
                errs = [os.path.basename(x) for x in fj["collection_error_files"]]
                lost = sum(1 for r in all_recs if r["cond"] == cond and r["proj"] == proj
                           and os.path.basename(r["file"]) in errs)
                suites[f"{cond}/{proj}"].update({"collected": fj["collected"], "rejected": fj["rejected"],
                                                 "collection_error_files": errs, "functions_in_error_files": lost})
    for r in all_recs:
        errs = suites.get(f"{r['cond']}/{r['proj']}", {}).get("collection_error_files", [])
        r["in_collection_error_file"] = int(os.path.basename(r["file"]) in errs)
    s = json.load(open(f"{R}/results/summary.json"))
    for k, v in s["suites"].items():
        if k in suites:
            suites[k]["collected_in_reruns"] = v["tests"]
    out_cases = case_studies(all_recs)
    per_cond = {CONDS[c]: aggregate([r for r in all_recs if r["cond"] == c]) for c in CONDS}
    per_cond["LLM (all)"] = aggregate([r for r in all_recs if r["cond"] != "human"])
    per_cond_lib = {f"{CONDS[c]}/{p}": aggregate([r for r in all_recs if r["cond"] == c and r["proj"] == p])
                    for c in CONDS for p in LIBS if any(r["cond"] == c and r["proj"] == p for r in all_recs)}
    per_cond_macro = {}
    for c in CONDS:
        libs = [v for k, v in per_cond_lib.items() if k.split("/")[0] == CONDS[c]]
        per_cond_macro[CONDS[c]] = {k: round(sum(v[k] for v in libs) / len(libs), 2) for k, _ in FEATURES}
        per_cond_macro[CONDS[c]]["libraries"] = len(libs)
    per_cond_no_arrow = {CONDS[c]: aggregate([r for r in all_recs if r["cond"] == c and r["proj"] != "arrow"]) for c in CONDS}
    for c in CONDS:
        ns = [v["n_test_functions"] for k, v in suites.items() if k.split("/")[0] == c]
        per_cond[CONDS[c]]["tests_per_suite"] = {"mean": round(sum(ns) / len(ns), 1), "min": min(ns), "max": max(ns),
                                                 "suites": len(ns), "over_80": sum(1 for x in ns if x > 80)}
    out = {"generated_by": "scripts/characterize.py", "features": dict(FEATURES), "per_condition": per_cond,
           "per_condition_macro": per_cond_macro, "per_condition_excluding_arrow": per_cond_no_arrow,
           "per_condition_library": per_cond_lib, "suites": suites, "flaky_matched": matched, "case_studies": out_cases, "tests": all_recs}
    json.dump(out, open(f"{R}/results/characterize.json", "w"), separators=(",", ":"))
    labs = list(per_cond)
    print("| Feature | " + " | ".join(labs) + " |")
    print("|---|" + "---|" * len(labs))
    print("| Test functions | " + " | ".join(str(per_cond[l]["n_tests"]) for l in labs) + " |")
    for k, name in FEATURES:
        print(f"| {name} | " + " | ".join(f"{per_cond[l][k]:.1f}" for l in labs) + " |")
    for k in NUMERIC:
        print(f"| {k} mean | " + " | ".join(f"{per_cond[l][k + '_mean']:.2f}" for l in labs) + " |")
    print("flaky matched:", sum(matched.values()), "of", len(matched))


if __name__ == "__main__":
    main()
