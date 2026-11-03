#!/usr/bin/env python3
"""Audit: did any Claude E2 matched-study generation subagent read a project test file?

Read-only. Scans every *.jsonl under ~/.claude/projects, finds transcripts that wrote
matched/<proj>/(sonnet|haiku)_plain_[12]/test_gen.py (Write/Edit/MultiEdit file_path, or a
Bash command that redirects/tees into that path), then inspects every tool call in those
transcripts for access to a test file of the project.

Usage: python3 -I audit_e2_claude_test_reads.py [--json out.json]
"""
import csv
import glob
import json
import os
import re
import shlex
import sys
from collections import defaultdict

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "ece382v-flaky-llm-tests")
PROJ_ROOT = os.path.join(HOME, ".claude", "projects")

SUITE_RE = re.compile(r"matched/([^/\s'\"]+)/((sonnet|haiku)_plain_([12]))/test_gen\.py")

# path-like token
PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_./~\-]+")


def is_test_path(p):
    """True if p looks like a test file of a project (not our own generated suite)."""
    p = p.strip().strip("'\"").rstrip(":,;)")
    if not p:
        return False
    if SUITE_RE.search(p) or "/matched/" in p and p.endswith("test_gen.py"):
        return False
    parts = [x for x in p.split("/") if x]
    if not parts:
        return False
    base = parts[-1]
    dirs = parts[:-1]
    if any(d in ("test", "tests", "testing") for d in dirs):
        return True
    if base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py":
        return True
    return False


def is_test_file_path(p):
    """Stricter: test path that names a file (has an extension), not just a directory."""
    p = p.strip().strip("'\"").rstrip(":,;)")
    base = p.rstrip("/").split("/")[-1]
    return is_test_path(p) and "." in base and not p.endswith("/")


def load_flaky():
    flaky = defaultdict(set)
    for fn in ("candidates.csv", "candidates_ext.csv"):
        path = os.path.join(REPO, "results", "gruber", fn)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                flaky[row["Project_Name"]].add(row["Test_filename"].strip())
    return flaky


def load_projects():
    with open(os.path.join(REPO, "results", "gruber", "matched_projects.txt")) as f:
        return f.read().split()


def iter_records(path):
    with open(path, errors="replace") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def content_blocks(rec):
    m = rec.get("message")
    if not isinstance(m, dict):
        return []
    c = m.get("content")
    return c if isinstance(c, list) else []


def result_text(block):
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


LIST_ONLY_CMDS = {"ls", "find", "mkdir", "wc", "tree", "stat", "file", "du", "test", "[", "echo", "pwd", "cd"}
READ_CMDS = {"cat", "head", "tail", "sed", "awk", "less", "more", "grep", "rg", "egrep", "fgrep",
             "python", "python3", "nl", "bat", "diff", "cp", "strings", "xxd", "od", "sort", "uniq", "cut"}


def bash_suite_writes(cmd):
    """Suites a bash command writes into (redirect / tee / python open w)."""
    out = set()
    for m in SUITE_RE.finditer(cmd):
        # redirect or tee directly before the path
        start = m.start()
        prefix = cmd[max(0, start - 200):start]
        if re.search(r"(>>?|tee(\s+-a)?)\s*['\"]?[^\s'\"]*$", prefix) or re.search(r"open\(\s*['\"][^'\"]*$", prefix):
            out.add((m.group(1), m.group(2)))
    return out


HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\s*(?=\n|$)", re.S)


def strip_heredocs(cmd):
    """Drop heredoc bodies (file content being written), keep the command line."""
    return HEREDOC_RE.sub(lambda m: m.group(0).split("\n", 1)[0], cmd)


def resolve(tok, cwd):
    tok = os.path.expanduser(tok)
    if not os.path.isabs(tok) and cwd:
        tok = os.path.normpath(os.path.join(cwd, tok))
    return tok


def expand(tok, cwd):
    """Resolve a token against cwd; expand shell globs against the current filesystem."""
    p = resolve(tok, cwd)
    if any(ch in p for ch in "*?["):
        got = glob.glob(p, recursive=True)
        return got if got else [p]
    return [p]


def classify_path(p, proj, own_suite_dir):
    """'test' for a project test file, 'other_suite' for another generated suite, else None."""
    if own_suite_dir and p.startswith(own_suite_dir):
        return None
    if "/matched/" in p and "/matched_src/" not in p:
        return "other_suite" if p.rstrip("/").endswith(".py") else None
    return "test" if is_test_path(p) else None


def analyze_bash(cmd, result, proj=None, own_suite_dir=None):
    """Return list of (path, kind) for test files touched by a bash command."""
    hits = []
    cmd = strip_heredocs(cmd)
    # drop quoted perl/python -e / -c program bodies: they edit the agent's own file
    cmd_nobody = re.sub(r"(perl|python3?|sed)\s+(-\S+\s+)*'[^']*'", r"\1 <prog>", cmd)
    cwd = None
    segments = re.split(r"\|\||&&|;|\||\n", cmd_nobody)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = seg.split()
        if not toks:
            continue
        prog = os.path.basename(toks[0])
        if prog == "cd" and len(toks) > 1:
            cwd = resolve(toks[1], cwd)
            continue
        cand = []
        for t in toks[1:]:
            if t.startswith("-") or not ("/" in t or t.endswith(".py") or "*" in t):
                continue
            for p in expand(t, cwd):
                k = classify_path(p, proj, own_suite_dir)
                if k:
                    cand.append((p, k))
        if not cand:
            continue
        if prog in LIST_ONLY_CMDS:
            if prog == "find" and re.search(r"-exec\s+(cat|grep|head|sed|tail)", seg):
                hits += [(p, "bash-find-exec-read") for p, k in cand]
            elif prog == "wc":
                hits += [(p, "bash-wc(count only)") for p, k in cand]
            continue
        kind = "bash-" + prog if prog in READ_CMDS else "bash-other:" + prog
        hits += [(p, kind + ("" if k == "test" else "[other_suite]")) for p, k in cand]
    # grep output that includes lines from test files ("path:line" or "path-line")
    if re.search(r"\b(grep|rg)\b", cmd) and result:
        for line in result.splitlines():
            m = re.match(r"^([^\s:]+\.py)[:\-](\d+[:\-])?", line)
            if m:
                p = resolve(m.group(1), cwd)
                if classify_path(p, proj, own_suite_dir) == "test" and is_test_file_path(p):
                    hits.append((p, "bash-grep-output"))
                continue
            # grep -l / -c style output: file names only, but grep searched inside the file
            m2 = re.match(r"^(\S+\.py)$", line.strip())
            if m2 and re.search(r"\bgrep\s+-\w*[rR]", cmd):
                p = resolve(m2.group(1), cwd)
                if classify_path(p, proj, own_suite_dir) == "test":
                    hits.append((p, "BORDERLINE grep -rl searched test contents, names only"))
    # cat/head output that contains "==> path <==" headers for test files
    for m in re.finditer(r"==> (\S+) <==", result or ""):
        p = resolve(m.group(1), cwd)
        if classify_path(p, proj, own_suite_dir) == "test":
            hits.append((p, "bash-output-header"))
    return hits


def analyze_grep_tool(inp, result):
    hits = []
    mode = inp.get("output_mode", "files_with_matches")
    p = inp.get("path", "") or ""
    if mode == "content":
        for line in (result or "").splitlines():
            m = re.match(r"^([^\s:]+\.py)[:\-]", line)
            if m and is_test_file_path(m.group(1)):
                hits.append((m.group(1), "Grep-content"))
        if is_test_file_path(p) and result and "No matches" not in result:
            hits.append((p, "Grep-content"))
    return hits


def main():
    projects = load_projects()
    flaky = load_flaky()
    files = sorted(glob.glob(os.path.join(PROJ_ROOT, "**", "*.jsonl"), recursive=True))

    suites = defaultdict(list)  # (proj, suite) -> list of transcript info
    for path in files:
        try:
            with open(path, errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        if not SUITE_RE.search(raw):
            continue
        written = set()
        models = set()
        tool_uses = []  # (id, name, input)
        results = {}
        first_prompt = None
        for rec in iter_records(path):
            t = rec.get("type")
            if t not in ("assistant", "user"):
                continue
            m = rec.get("message", {})
            if t == "assistant" and isinstance(m, dict) and m.get("model"):
                models.add(m["model"])
            if t == "user" and first_prompt is None:
                c = m.get("content") if isinstance(m, dict) else None
                first_prompt = c if isinstance(c, str) else " ".join(
                    b.get("text", "") for b in (c or []) if isinstance(b, dict) and b.get("type") == "text")
            for b in content_blocks(rec):
                if t == "assistant" and b.get("type") == "tool_use":
                    name, inp = b.get("name"), b.get("input") or {}
                    tool_uses.append((b.get("id"), name, inp))
                    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
                        fp = inp.get("file_path", "") or ""
                        mm = SUITE_RE.search(fp)
                        if mm:
                            written.add((mm.group(1), mm.group(2)))
                    elif name == "Bash":
                        written |= bash_suite_writes(inp.get("command", ""))
                elif t == "user" and b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = result_text(b)
        if not written:
            continue
        is_sub = "/subagents/" in path
        # analyse reads
        reads = []
        cmds = []
        proj0, suite0 = next(iter(written))
        own_dir = os.path.join(REPO, "matched", proj0, suite0) + "/"
        for tid, name, inp in tool_uses:
            res = results.get(tid, "")
            if name == "Read":
                fp = inp.get("file_path", "")
                k = classify_path(fp, proj0, own_dir)
                if k:
                    reads.append((fp, "Read" + ("" if k == "test" else "[other_suite]")))
            elif name == "Bash":
                cmd = inp.get("command", "")
                cmds.append(cmd[:160].replace("\n", " "))
                reads += analyze_bash(cmd, res, proj0, own_dir)
            elif name == "Grep":
                reads += analyze_grep_tool(inp, res)
            elif name in ("Glob", "LS"):
                pass  # names only
        meta = {}
        mp = path[:-6] + ".meta.json"
        if os.path.exists(mp):
            try:
                meta = json.load(open(mp))
            except Exception:
                pass
        for proj, suite in written:
            suites[(proj, suite)].append({
                "transcript": path, "subagent": is_sub, "models": sorted(models),
                "meta_model": meta.get("model"), "meta_desc": meta.get("description"),
                "n_suites_written": len(written),
                "reads": reads if len(written) == 1 else reads,  # multi-suite transcripts flagged below
                "bash_cmds": cmds,
            })

    expected = [(p, f"{m}_plain_{k}") for p in projects for m in ("sonnet", "haiku") for k in (1, 2)]
    found = [s for s in expected if any(x["subagent"] and x["n_suites_written"] == 1 for x in suites[s])]
    missing = [s for s in expected if s not in found]

    report = {"expected": len(expected), "found": len(found), "missing": missing, "suites": {}}
    model_ids = defaultdict(set)
    zero, anyread = [], []
    for s in expected:
        gens = [x for x in suites.get(s, []) if x["subagent"] and x["n_suites_written"] == 1]
        others = [x for x in suites.get(s, []) if not (x["subagent"] and x["n_suites_written"] == 1)]
        reads = []
        for g in gens:
            for mid in g["models"]:
                model_ids[s[1].split("_")[0]].add(mid)
            for fp, kind in g["reads"]:
                rel = fp.split(f"matched_src/{s[0]}/")[-1] if f"matched_src/{s[0]}/" in fp else fp
                reads.append({"path": fp, "tool": kind, "is_flaky_file": rel in flaky.get(s[0], set()),
                              "transcript": os.path.basename(g["transcript"])})
        report["suites"]["/".join(s)] = {
            "generators": [{k: g[k] for k in ("transcript", "models", "meta_model", "meta_desc", "bash_cmds")} for g in gens],
            "other_transcripts_writing": [o["transcript"] for o in others],
            "test_reads": reads,
            "flaky_files": sorted(flaky.get(s[0], set())),
        }
        if gens:
            (anyread if reads else zero).append(s)

    print(f"Expected suites: {len(expected)}; generating subagent transcript located: {len(found)}")
    if missing:
        print("Missing:", ", ".join("/".join(m) for m in missing))
    multi = {k: v for k, v in report["suites"].items() if len(v["generators"]) > 1}
    print(f"Suites with more than one generating subagent transcript: {len(multi)}")
    for k, v in multi.items():
        print("  ", k, [os.path.basename(g["transcript"]) for g in v["generators"]])
    print(f"Suites with zero test-file reads: {len(zero)}; with any: {len(anyread)}")
    for s in anyread:
        r = report["suites"]["/".join(s)]
        for x in r["test_reads"]:
            print(f"  {'/'.join(s)}: {x['tool']} {x['path']} flaky_file={x['is_flaky_file']} ({x['transcript']})")
    print("Model ids:", {k: sorted(v) for k, v in model_ids.items()})
    # bash usage summary
    nb = {k: v for k, v in report["suites"].items() if any(g["bash_cmds"] for g in v["generators"])}
    print(f"Suites whose generator ran any Bash command: {len(nb)}")
    for k, v in nb.items():
        for g in v["generators"]:
            for c in g["bash_cmds"]:
                print(f"  {k}: {c}")
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        with open(out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
