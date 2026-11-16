# -*- coding: utf-8 -*-
"""Unit tests for TimeMarker as exposed by the timemarker package.

The package __init__ asserts a minimum Python version and re-exports
TimeMarker from timemarker.timer. The version check runs at import time, so
those checks run in a fresh interpreter. The subprocess receives a modified
sys.version_info only in its own process, so this test process is not changed.
"""
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import timemarker


def _find_project_root():
    """Walk up from this file until the directory containing the package is found."""
    here = Path(__file__).resolve()
    for candidate in [here.parent] + list(here.parents):
        if (candidate / "timemarker" / "__init__.py").is_file():
            return candidate
    raise RuntimeError("could not locate the timemarker project root")


PROJECT_ROOT = _find_project_root()
INIT_FILE = PROJECT_ROOT / "timemarker" / "__init__.py"
ASSERT_MESSAGE = "Currently only python >= 3.6 is supported"


def _run_python(snippet, fake_version=None):
    """Run a snippet in a fresh interpreter, optionally with a faked sys.version_info."""
    prefix = ""
    if fake_version is not None:
        prefix = "import sys; sys.version_info = {!r}; ".format(tuple(fake_version))
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT)] + ([existing] if existing else [])
    )
    return subprocess.run(
        [sys.executable, "-c", prefix + snippet],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=120,
    )


def test_timemarker_is_exposed_on_package():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_attribute_is_a_class():
    assert inspect.isclass(timemarker.TimeMarker)


def test_timemarker_is_callable():
    assert callable(timemarker.TimeMarker)


def test_timemarker_name_is_TimeMarker():
    assert timemarker.TimeMarker.__name__ == "TimeMarker"


def test_timemarker_is_defined_in_timer_module():
    timer = importlib.import_module("timemarker.timer")
    assert timemarker.TimeMarker is timer.TimeMarker


def test_timemarker_class_module_is_timemarker_timer():
    assert timemarker.TimeMarker.__module__ == "timemarker.timer"


def test_timemarker_is_in_package_namespace_dict():
    assert "TimeMarker" in vars(timemarker)


def test_repeated_attribute_access_returns_same_class():
    first = timemarker.TimeMarker
    second = timemarker.TimeMarker
    assert first is second


def test_reimporting_package_returns_same_class_object():
    import timemarker as again

    assert again is timemarker
    assert sys.modules["timemarker"].TimeMarker is timemarker.TimeMarker


def test_package_file_is_init_module():
    assert Path(timemarker.__file__).resolve() == INIT_FILE.resolve()


def test_init_source_declares_utf8_encoding():
    first_line = INIT_FILE.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "# -*- coding: utf-8 -*-"


def test_init_source_contains_version_assertion_with_message():
    text = INIT_FILE.read_text(encoding="utf-8")
    assert "assert version_info >= (3, 6)" in text
    assert ASSERT_MESSAGE in text


def test_import_succeeds_on_current_interpreter():
    result = _run_python("import timemarker; assert timemarker.TimeMarker is not None")
    assert result.returncode == 0, result.stderr


def test_import_succeeds_at_minimum_supported_version():
    result = _run_python(
        "import timemarker; assert timemarker.TimeMarker is not None",
        fake_version=(3, 6, 0),
    )
    assert result.returncode == 0, result.stderr


def test_import_fails_just_below_minimum_version():
    result = _run_python("import timemarker", fake_version=(3, 5, 10))
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert ASSERT_MESSAGE in result.stderr


def test_import_fails_on_python_2():
    result = _run_python("import timemarker", fake_version=(2, 7, 18))
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert ASSERT_MESSAGE in result.stderr


def test_import_fails_on_very_old_python_3():
    result = _run_python("import timemarker", fake_version=(3, 0, 0))
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert ASSERT_MESSAGE in result.stderr


def test_timemarker_available_after_successful_import_in_subprocess():
    result = _run_python(
        "import timemarker, sys; "
        "sys.stdout.write('ok' if hasattr(timemarker, 'TimeMarker') else 'missing')",
        fake_version=(3, 8, 0),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"
