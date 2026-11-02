#!/usr/bin/env python3
"""Generate test suites with a second model family (OpenAI) using the same plain prompt as PL-S.

OpenAI models are called through the API, so they cannot read files themselves. The prompt is the plain
prompt from prompts/README.md with the source files pasted in, one shot, no execution. The key is read
from OPENAI_API_KEY or ~/Library/Application Support/PageCatcher/openai-key.txt and never written anywhere.

  python scripts/gen_openai.py matched <Project_Name> <model> <sample>     -> matched/<project>/<tag>_plain_<k>/test_gen.py
  python scripts/gen_openai.py matched <Project_Name> <model> <sample> guard
                                                                          -> matched/<project>/<tag>_guard_<k>/test_gen.py
     (plain prompt plus the GUARD list of flakiness pitfalls, the mitigation prompt of RQ4; variant guard2
      adds GUARD2, the three mechanisms found in the guard trial)
  python scripts/gen_openai.py library <lib> <model> <sample>              -> gen_plain_<tag>/<lib>/test_p<tag>_<lib>.py
Every call is logged to results/openai_calls.jsonl (model, target, tokens, seconds), without the key.
"""
import json, os, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
TAGS = {"gpt-5.4-mini": "gpt54mini", "gpt-4.1-mini": "gpt41mini"}


GUARD = ("These tests must not be flaky: every test must give the same result in every run, in any order, on any "
         "machine. In particular: if the code under test uses random numbers (random, numpy, torch), seed every "
         "generator in the test or assert only properties that hold for every possible random outcome; do not depend on "
         "the wall clock, sleep, or how long something takes; do not depend on the order tests run in, and undo any "
         "change to module level or global state; do not depend on the order of sets or dict keys unless the code "
         "guarantees it; compare floating point results with a tolerance. ")


# Guard prompt v2 (Nov 3): GUARD plus the three mechanisms behind the guard trial's flaky tests
# (results/root_causes/labels.json): the salted builtin hash(), seeding in a fixture that pytest-randomly
# later overrides, and a module attribute replaced by hand and never restored.
GUARD2 = GUARD + ("Three cases need extra care. First, seed random, numpy.random and torch at the start of the test "
                  "function body itself, not in a fixture, whenever the code under test might draw random numbers, even "
                  "if the test does not call them directly. Second, Python's builtin hash() of str and bytes changes in "
                  "every run, so if the code uses hash(), for example to pick buckets or slots, assert only results that "
                  "hold for every hash value, or use int keys, whose hash is fixed. Third, replace module attributes "
                  "only with pytest's monkeypatch fixture, so they are restored even when the test fails. ")


def key():
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        k = open(os.path.expanduser("~/Library/Application Support/PageCatcher/openai-key.txt")).read().strip()
    return k


def call(model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if model.startswith("gpt-5"):
        body["reasoning_effort"] = "low"
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    return j["choices"][0]["message"]["content"], j.get("usage", {}), round(time.time() - t0, 1)


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def files_block(base, rels):
    out = []
    for rel in rels:
        p = os.path.join(base, rel)
        paths = [p] if os.path.isfile(p) else sorted(
            os.path.join(dp, f) for dp, _, fs in os.walk(p) for f in fs
            if f.endswith(".py") and "/tests/" not in dp + "/" and not f.startswith("test"))
        for f in paths:
            out.append(f"### File: {os.path.relpath(f, base)}\n```python\n{open(f, encoding='utf-8', errors='replace').read()}\n```")
    return "\n\n".join(out)


def main():
    mode, target, model, sample = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    variant = sys.argv[5] if len(sys.argv) > 5 else "plain"
    tag = TAGS[model]
    if mode == "matched":
        T = json.load(open(f"{ROOT}/matched/targets.json"))[target]
        base = f"{ROOT}/matched_src/{target}"
        prompt = (f"Write pytest unit tests for the Python project {target}. Focus on this code: {T['names']} in "
                  f"{', '.join(T['files'])}. The source of those files is below. Write between 10 and 25 test functions in "
                  f"one file. The tests will run with Python 3.8 and pytest 7, with the project and its own dependencies "
                  f"installed, from the project root; do not use other third party packages. Reply with only the test file "
                  f"in one python code block.\n\n")
        if variant in ("guard", "guard2"):
            prompt = prompt.replace("Reply with only", (GUARD if variant == "guard" else GUARD2) + "Reply with only")
        prompt += files_block(base, T["files"])
        out = f"{ROOT}/matched/{target}/{tag}_{variant}_{sample}/test_gen.py"
    else:
        from build_testpilot_context import MODULES
        sub, rels = MODULES[target]
        base = f"{ROOT}/subjects/{sub}"
        prompt = (f"Write unit tests for the Python library {target} using pytest. The library source is below. "
                  f"Write between 50 and 80 test functions in one file. Only pytest, pytest-asyncio, freezegun and the "
                  f"Python standard library are installed besides the library itself. Reply with only the test file in "
                  f"one python code block.\n\n" + files_block(base, rels))
        out = f"{ROOT}/gen_plain_{tag}/{target}/test_p{tag}_{target}.py"
    text, usage, secs = call(model, prompt)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(extract_code(text))
    with open(f"{ROOT}/results/openai_calls.jsonl", "a") as f:
        f.write(json.dumps({"mode": mode, "target": target, "model": model, "sample": sample, "variant": variant, "out": os.path.relpath(out, ROOT),
                            "prompt_chars": len(prompt), "usage": usage, "seconds": secs}) + "\n")
    n = len(re.findall(r"^\s*(?:async )?def test_", open(out).read(), re.M))
    print(f"{mode} {target} {model} #{sample}: {n} tests, {usage.get('prompt_tokens')}+{usage.get('completion_tokens')} tokens, {secs}s")


if __name__ == "__main__":
    main()
