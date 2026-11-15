#!/usr/bin/env python3
"""Test quality of the matched study suites: line coverage and mutation score on the code under test.

For one matched study project: set it up the same way as matched_run.py, then for every suite that matches --suites
  coverage:  run the suite once in file order (PYTHONHASHSEED=0) with pytest-cov and report the share of statements
             covered in the target files of matched/targets.json, the files every model was asked to test.
  mutation:  build one fixed set of mutants of the target files (seeded by the project name, at most --mutants), the
             same set for every suite of the project, and run the suite once per mutant. A mutant is killed when a
             test that passed on the original code fails, errors, or times out. Mutants are single token edits made on
             the source text (comparison and arithmetic operators, and/or, numeric constants plus one, True/False,
             return values replaced by None), so the rest of the file keeps its formatting.

  python scripts/e2_quality.py <Project_Name> --suites '*_plain_*,gpt41mini_guard*' --out results/quality/<p>.json

Output JSON: project, mutants: [{file, line, kind}], suites: {sid: {tests, passing, covered, statements, coverage,
mutants, killed, timeouts, score}}. Mutant runs use the passing tests of the clean run only.
"""
import argparse, ast, glob, json, os, random, shutil, sys, tempfile, time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gruber_reproduce as gr

ROOT = gr.ROOT
CMP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"), ast.Eq: ("==", "!="),
       ast.NotEq: ("!=", "==")}
BIN = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "/"), ast.Div: ("/", "*"), ast.FloorDiv: ("//", "/"),
       ast.Mod: ("%", "*")}


def _op_between(lines, a, b, old):
    # position of the operator text `old` between node a (end) and node b (start), on one line only
    if a.end_lineno != b.lineno:
        return None
    seg = lines[a.end_lineno - 1][a.end_col_offset:b.col_offset]
    i = seg.find(old)
    if i < 0 or (old in ("<", ">") and seg[i:i + 2] in ("<=", ">=", "<<", ">>")):
        return None
    return (a.end_lineno, a.end_col_offset + i, a.end_col_offset + i + len(old))


def mutants_for(path, names):
    src = open(path, encoding="utf-8", errors="replace").read()
    lines = src.split("\n")
    tree = ast.parse(src)
    wanted = {n.strip() for n in names.split(",") if n.strip()}
    roots = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in wanted]
    roots = roots or [tree]
    out, seen = [], set()
    for root in roots:
        for n in ast.walk(root):
            m = None
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in CMP:
                old, new = CMP[type(n.ops[0])]
                pos = _op_between(lines, n.left, n.comparators[0], old)
                m = pos and ("compare", pos, new)
            elif isinstance(n, ast.BinOp) and type(n.op) in BIN:
                old, new = BIN[type(n.op)]
                pos = _op_between(lines, n.left, n.right, old)
                m = pos and ("arith", pos, new)
            elif isinstance(n, ast.BoolOp) and len(n.values) == 2:
                old, new = ("and", "or") if isinstance(n.op, ast.And) else ("or", "and")
                pos = _op_between(lines, n.values[0], n.values[1], old)
                m = pos and ("bool", pos, new)
            elif isinstance(n, ast.Constant) and n.lineno == n.end_lineno and type(n.value) in (int, float, bool):
                if isinstance(n.value, bool):
                    new = "False" if n.value else "True"
                else:
                    new = repr(n.value + 1)
                m = ("constant", (n.lineno, n.col_offset, n.end_col_offset), new)
            elif isinstance(n, ast.Return) and n.value is not None and n.value.lineno == n.value.end_lineno \
                    and not (isinstance(n.value, ast.Constant) and n.value.value is None):
                v = n.value
                m = ("return", (v.lineno, v.col_offset, v.end_col_offset), "None")
            if m and m[1] not in seen:
                seen.add(m[1]); out.append(m)
    return out


def apply(path, mutant):
    kind, (ln, a, b), new = mutant
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    lines[ln - 1] = lines[ln - 1][:a] + new + lines[ln - 1][b:]
    text = "\n".join(lines)
    ast.parse(text)
    open(path, "w", encoding="utf-8").write(text)


def pytest(py, repo, target, extra, timeout_s=60, wall=600):
    fd, xml = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    path = os.pathsep.join([repo] + ([f"{repo}/src"] if os.path.isdir(f"{repo}/src") else []))
    env = dict(os.environ, PYTHONPATH=path, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", "--continue-on-collection-errors",
           f"--timeout={timeout_s}", "-o", "python_files=test_*.py", "-o", "addopts=", "--junitxml", xml] + extra + target
    rc, out = gr.sh(cmd, cwd=repo, timeout=wall, env=env)
    res = {}
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            st = "pass"
            for ch in tc:
                if ch.tag in ("failure", "error", "skipped"):
                    st = {"failure": "fail", "error": "error", "skipped": "skip"}[ch.tag]
            res[f'{tc.get("classname")}::{tc.get("name")}'] = st
    except Exception:
        pass
    finally:
        os.unlink(xml)
    return rc, res


def node_ids(sid, tids):
    # junit "zz_sid.test_gen::name" or "zz_sid.test_gen.Class::name" -> pytest node ids
    out = []
    for t in tids:
        cls, name = t.split("::", 1)
        parts = cls.split(".")
        out.append("::".join(["/".join(parts[:2]) + ".py"] + parts[2:] + [name]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project")
    ap.add_argument("--work", default="/tmp/quality_work"); ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="*_plain_*"); ap.add_argument("--mutants", type=int, default=40)
    a = ap.parse_args()
    r = gr.row_for(a.project); t0 = time.time()
    T = json.load(open(f"{ROOT}/matched/targets.json"))[a.project]
    d = f"{a.work}/{a.project}"; repo = f"{d}/repo"; venv = f"{d}/venv"
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    gr.sh(["git", "init", "-q", repo]); gr.sh(["git", "-C", repo, "remote", "add", "origin", r["Project_URL"]])
    gr.sh(["git", "-C", repo, "fetch", "-q", "--depth", "1", "origin", r["Project_Hash"]], env=env)
    gr.sh(["git", "-C", repo, "checkout", "-q", "FETCH_HEAD"])
    log = []
    ok, py = gr.install(repo, venv, log)
    gr.sh(["uv", "pip", "install", "-q", "-p", py, "pytest-randomly<3.16", "freezegun", "pytest-mock", "pytest-cov<5", "coverage<7.6"])
    gr.fix_missing_modules(py, repo, r["Test_filename"], log)
    files = [f for f in T["files"] if os.path.isfile(f"{repo}/{f}")]
    rng = random.Random(a.project)
    allm = [(f, m) for f in files for m in mutants_for(f"{repo}/{f}", T["names"])]
    picked = sorted(rng.sample(allm, min(a.mutants, len(allm))), key=lambda x: (x[0], x[1][1]))
    res = {"project": a.project, "install_ok": ok, "target_files": files, "mutant_pool": len(allm),
           "mutants": [{"file": f, "line": m[1][0], "kind": m[0]} for f, m in picked], "suites": {}}
    pats = [p for p in a.suites.split(",") if p]
    sdirs = sorted({x for p in pats for x in glob.glob(f"{ROOT}/matched/{a.project}/{p}/")})
    for sdir in sdirs:
        sid = os.path.basename(sdir.rstrip("/"))
        if sid.split("_")[1] in ("plainseed", "plainiso"):
            continue  # same tests as plain
        dst = f"{repo}/zz_{sid}"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
        shutil.copy(f"{sdir}/test_gen.py", dst)
        cov_json = f"{d}/cov_{sid}.json"
        _, clean = pytest(py, repo, [f"zz_{sid}"], ["--cov=.", f"--cov-report=json:{cov_json}", "--cov-config=/dev/null"], wall=900)
        passing = sorted(t for t, s in clean.items() if s == "pass")
        covered = stmts = 0
        if os.path.exists(cov_json):
            cj = json.load(open(cov_json))["files"]
            for f in files:
                v = next((cj[k] for k in cj if os.path.normpath(k).endswith(os.path.normpath(f))), None)
                if v:
                    covered += v["summary"]["covered_lines"]; stmts += v["summary"]["num_statements"]
        s = {"tests": len(clean), "passing": len(passing), "covered": covered, "statements": stmts,
             "coverage": round(100 * covered / stmts, 1) if stmts else None, "mutants": 0, "killed": 0, "timeouts": 0}
        if passing:
            ids = node_ids(sid, passing)
            for f, m in picked:
                orig = open(f"{repo}/{f}", encoding="utf-8", errors="replace").read()
                try:
                    apply(f"{repo}/{f}", m)
                except SyntaxError:
                    continue
                try:
                    rc, mres = pytest(py, repo, ids, [], timeout_s=30, wall=300)
                finally:
                    open(f"{repo}/{f}", "w", encoding="utf-8").write(orig)
                s["mutants"] += 1
                if rc == -9:
                    s["killed"] += 1; s["timeouts"] += 1
                elif any(mres.get(t) in ("fail", "error") for t in passing) or not mres:
                    s["killed"] += 1
        s["score"] = round(100 * s["killed"] / s["mutants"], 1) if s["mutants"] else None
        res["suites"][sid] = s
        print(f"[{a.project}/{sid}] passing={s['passing']} coverage={s['coverage']} mutation={s['killed']}/{s['mutants']}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
