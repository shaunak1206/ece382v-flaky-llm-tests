"""Mitigation M1: seed every random number generator right before each test body runs.

Added after generation to a suite's folder as conftest.py. The seeding happens inside the test call, not in a
fixture, because pytest-randomly reseeds random and numpy in pytest_runtest_call, after fixtures have run, and
that undoes a seeding fixture (the gpt41mini_guard_1 pylabeledrf test failed this way). It cannot fix tests that
depend on the builtin hash() of str or bytes, which Python salts per process (PYTHONHASHSEED).
"""
import functools
import random


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
