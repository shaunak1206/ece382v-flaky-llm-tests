"""Mitigation M2: M1 seeding plus restoring interpreter state after every test.

Added after generation to a suite's folder as conftest.py. M1 (seed_conftest.py) leaves the order dependent
tests that GPT 5.4 mini writes when it reloads a package or pops it from sys.modules: a later test then sees a
new module object, or a package whose attributes were rebound by importlib.reload while a monkeypatch was
active. This fixture snapshots, before each test's setup, every sys.modules entry, the globals of every loaded
module, sys.path, os.environ and the working directory, and puts back whatever changed after teardown. Modules
imported for the first time during a test stay loaded and bound on their parent package. pytest's own
modules, sys and builtins are skipped.
"""
import functools
import os
import random
import sys
import types

import pytest

SKIP = ("_pytest", "pytest", "pluggy", "py", "_hypothesis", "hypothesis", "sys", "builtins", "__main__")


def _seed():
    random.seed(0)
    try:
        import numpy
        numpy.random.seed(0)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(0)
    except ImportError:
        pass


def pytest_collection_modifyitems(items):
    for item in items:
        fn = getattr(item, "obj", None)
        if not callable(fn) or not hasattr(item, "funcargs"):
            continue

        @functools.wraps(fn)
        def seeded(*args, __fn=fn, **kwargs):
            _seed()
            return __fn(*args, **kwargs)

        item.obj = seeded


def _skip(name):
    return name.split(".")[0] in SKIP


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    modules = dict(sys.modules)
    globals_ = {n: dict(vars(m)) for n, m in modules.items() if m is not None and not _skip(n) and hasattr(m, "__dict__")}
    path, environ, cwd = list(sys.path), dict(os.environ), os.getcwd()
    yield
    for n, m in modules.items():
        if sys.modules.get(n) is not m:
            sys.modules[n] = m
    for n, snap in globals_.items():
        d = vars(modules[n])
        # a submodule imported during the test is bound on its parent package; keep that binding
        for k in [k for k in d if k not in snap and not isinstance(d[k], types.ModuleType)]:
            del d[k]
        for k, v in snap.items():
            if d.get(k, d) is not v:
                d[k] = v
    sys.path[:] = path
    if dict(os.environ) != environ:
        os.environ.clear(); os.environ.update(environ)
    if os.getcwd() != cwd:
        os.chdir(cwd)
