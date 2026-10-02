"""
Shared pytest fixtures for the LLM-generated `schedule` test suite.

These tests are written against the public API of the `schedule` package
found in subjects/schedule/schedule/__init__.py, based only on reading its
source code and docstrings.
"""
import pytest

import schedule


@pytest.fixture(autouse=True)
def clean_default_scheduler():
    """
    Several tests exercise the module-level convenience functions
    (schedule.every, schedule.run_pending, ...), which all operate on the
    shared `schedule.default_scheduler` instance. Clear it before and after
    every test so tests don't leak jobs into one another.
    """
    schedule.clear()
    yield
    schedule.clear()


@pytest.fixture
def scheduler():
    """A fresh, isolated Scheduler instance."""
    return schedule.Scheduler()
