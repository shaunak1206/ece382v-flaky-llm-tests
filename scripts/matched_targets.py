#!/usr/bin/env python3
"""Find the code under test of a known flaky test, so LLMs can be asked to test the SAME code.

For a Gruber et al. flaky test we parse its test file, find the test function, and collect the project
names it uses (functions, classes, modules imported from the project package). Those names and the files
that define them become the generation target. The generators never see the human test itself.

  python scripts/matched_targets.py <Project_Name> --repo <checkout dir>   (prints JSON, also used by gen)
Output JSON: project, test_file, test_func, package_dirs, targets (list of {name, module_file}),
target_files (unique source files to show the generator).
"""
import argparse, ast, csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def row_for(name):
    for f in ("candidates.csv", "candidates_ext.csv"):
        for r in csv.DictReader(open(f"{ROOT}/results/gruber/{f}")):
            if r["Project_Name"] == name:
                return r
    raise SystemExit(f"unknown project {name}")


def is_test_path(p):
    b = os.path.basename(p)
    return "/test/" in p or "/tests/" in p or "/testing/" in p or b.startswith("test_") or b.endswith("_test.py") or b == "conftest.py"


def package_dirs(repo):
    out = []
    for base in (repo, f"{repo}/src", f"{repo}/lib", f"{repo}/Lib"):
        if not os.path.isdir(base): continue
        for n in os.listdir(base):
            p = f"{base}/{n}"
            if os.path.isdir(p) and os.path.exists(f"{p}/__init__.py") and not is_test_path(p + "/"):
                out.append(p)
            elif n.endswith(".py") and not is_test_path(p) and n not in ("setup.py", "conftest.py"):
                out.append(p)
    return out


def module_file(repo, modname, pkgs):
    parts = modname.split(".")
    for base in (repo, f"{repo}/src", f"{repo}/lib", f"{repo}/Lib"):
        cand = os.path.join(base, *parts)
        if os.path.exists(cand + ".py"): return cand + ".py"
        if os.path.exists(cand + "/__init__.py"): return cand + "/__init__.py"
    return None


def find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project"); ap.add_argument("--repo", required=True)
    a = ap.parse_args(); r = row_for(a.project); repo = a.repo
    tf = f"{repo}/{r['Test_filename']}"
    tree = ast.parse(open(tf, encoding="utf-8", errors="replace").read())
    pkgs = package_dirs(repo)
    pkg_names = {os.path.basename(p).removesuffix(".py") for p in pkgs}
    imported = {}  # local name -> (module, attr or None)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] in pkg_names:
                for al in node.names:
                    imported[al.asname or al.name] = (node.module, al.name)
        elif isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[0] in pkg_names:
                    imported[(al.asname or al.name).split(".")[0]] = (al.name, None)
    fn = find_func(tree, r["Test_funcname"])
    used = set()
    scope = fn if fn is not None else tree
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and node.id in imported: used.add(node.id)
        if isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute): base = base.value
            if isinstance(base, ast.Name) and base.id in imported: used.add(base.id)
    if not used:  # fall back to everything the test module imports from the project (setup methods, fixtures)
        used = set(imported)
    targets, files = [], []
    for u in sorted(used):
        mod, attr = imported[u]
        f = module_file(repo, mod, pkgs)
        if f is None and attr:  # "from pkg import submodule"
            f = module_file(repo, f"{mod}.{attr}", pkgs)
        targets.append({"name": attr or mod, "module": mod, "module_file": os.path.relpath(f, repo) if f else None})
        if f and os.path.relpath(f, repo) not in files: files.append(os.path.relpath(f, repo))
    print(json.dumps({"project": a.project, "test_file": r["Test_filename"], "test_func": r["Test_funcname"],
                      "category": r["flakiness category"], "package_dirs": [os.path.relpath(p, repo) for p in pkgs],
                      "targets": targets, "target_files": files}, indent=1))


if __name__ == "__main__":
    main()
