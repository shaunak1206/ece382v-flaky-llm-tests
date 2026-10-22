#!/usr/bin/env python3
"""Build the TestPilot style prompt context for each subject.

TestPilot (Schaefer et al., TSE 2024) prompts the model with a function's signature, its doc
comment, and usage snippets mined from the documentation, not the whole implementation. We adapt
that to a module level prompt: every public function, class, and method of the modules under
test, with its signature and docstring, plus code examples taken from the README.
Writes prompts/context_testpilot/<proj>.md
"""
import ast, os, re, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# proj -> (subject dir, module files or dirs relative to it), same modules as the original API descriptions
MODULES = {
 "cachetools": ("cachetools", ["src/cachetools"]),
 "schedule": ("schedule", ["schedule/__init__.py"]),
 "tinydb": ("tinydb", ["tinydb"]),
 "humanize": ("humanize", ["src/humanize"]),
 "toolz": ("toolz", ["toolz/itertoolz.py", "toolz/functoolz.py", "toolz/dicttoolz.py", "toolz/recipes.py", "toolz/sandbox"]),
 "tabulate": ("python-tabulate", ["tabulate/__init__.py"]),
 "tenacity": ("tenacity", ["tenacity"]),
 "filelock": ("filelock", ["src/filelock"]),
 "boltons": ("boltons", [f"boltons/{m}.py" for m in ["cacheutils", "iterutils", "timeutils", "queueutils", "fileutils", "dictutils", "setutils", "strutils"]]),
 "more_itertools": ("more-itertools", ["more_itertools/more.py", "more_itertools/recipes.py"]),
 "arrow": ("arrow", [f"arrow/{m}.py" for m in ["api", "factory", "arrow", "parser", "formatter"]]),
 "pybreaker": ("pybreaker", ["src/pybreaker/__init__.py"]),
}
DOC_CAP = 700

def files_for(base, rel):
    p = os.path.join(base, rel)
    if os.path.isdir(p):
        return sorted(f for f in glob.glob(p + "/**/*.py", recursive=True)
                      if "/tests/" not in f and "/test_" not in f and not os.path.basename(f).startswith("test"))
    return [p]

def sig(node):
    a = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    pre = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{pre} {node.name}({a}){ret}"

def doc(node, cap=DOC_CAP):
    d = ast.get_docstring(node) or ""
    d = d.strip()
    return d if len(d) <= cap else d[:cap].rstrip() + " ..."

def module_api(path, modname):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    allnames = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            try: allnames = set(ast.literal_eval(n.value))
            except Exception: pass
    public = lambda name: (name in allnames) if allnames is not None else not name.startswith("_")
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and public(n.name):
            out.append(f"```python\n{sig(n)}\n```\n{doc(n)}\n")
        elif isinstance(n, ast.ClassDef) and public(n.name):
            bases = ", ".join(ast.unparse(b) for b in n.bases)
            out.append(f"```python\nclass {n.name}({bases})\n```\n{doc(n)}\n")
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not m.name.startswith("_") or m.name in ("__init__", "__call__", "__enter__", "__exit__")):
                    out.append(f"```python\n# method of {n.name}\n{sig(m)}\n```\n{doc(m, 400)}\n")
    return out

def readme_examples(base, cap=6000):
    for name in ["README.rst", "README.md"]:
        p = os.path.join(base, name)
        if not os.path.exists(p): continue
        txt = open(p, encoding="utf-8").read()
        blocks = re.findall(r"```(?:python|py|pycon)?\n(.*?)```", txt, re.S)
        if not blocks:  # rst: indented blocks after '::' or code-block directives
            blocks = [re.sub(r"^ {2,4}", "", b, flags=re.M) for b in re.findall(r"(?:::|code-block:: \w+)\s*\n\n((?:[ ]{2,}.*\n|\n)+)", txt)]
        blocks = [b.strip() for b in blocks if b.strip() and ("import" in b or ">>>" in b or "(" in b)]
        out, n = [], 0
        for b in blocks:
            if n + len(b) > cap: break
            out.append("```python\n" + b + "\n```"); n += len(b)
        return out
    return []

def main():
    os.makedirs(f"{ROOT}/prompts/context_testpilot", exist_ok=True)
    for proj, (sub, rels) in MODULES.items():
        base = f"{ROOT}/subjects/{sub}"
        parts = [f"# {proj}: public API (signatures and documentation)\n"]
        for rel in rels:
            for f in files_for(base, rel):
                mod = os.path.relpath(f, base)
                api = module_api(f, mod)
                if api:
                    parts.append(f"\n## module `{mod}`\n")
                    parts += api
        ex = readme_examples(base)
        if ex:
            parts.append("\n# Usage examples from the project README\n")
            parts += ex
        text = "\n".join(parts)
        open(f"{ROOT}/prompts/context_testpilot/{proj}.md", "w").write(text)
        print(f"{proj:16s} {len(text):7d} chars")

if __name__ == "__main__":
    main()
