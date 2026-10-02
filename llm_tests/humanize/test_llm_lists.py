"""Unit tests for humanize.lists.natural_list.

Based only on the public function signature and docstring in
subjects/humanize/src/humanize/lists.py.
"""

import pytest

from humanize import lists


def test_natural_list_three_items():
    assert lists.natural_list(["one", "two", "three"]) == "one, two and three"


def test_natural_list_two_items():
    assert lists.natural_list(["one", "two"]) == "one and two"


def test_natural_list_single_item():
    assert lists.natural_list(["one"]) == "one"


def test_natural_list_empty_returns_empty_string():
    assert lists.natural_list([]) == ""


def test_natural_list_many_items():
    items = ["a", "b", "c", "d", "e"]
    assert lists.natural_list(items) == "a, b, c, d and e"


def test_natural_list_non_string_items_are_stringified():
    assert lists.natural_list([1, 2, 3]) == "1, 2 and 3"


def test_natural_list_single_non_string_item():
    assert lists.natural_list([42]) == "42"


def test_natural_list_does_not_use_oxford_comma():
    result = lists.natural_list(["red", "green", "blue"])
    assert ", blue" not in result
    assert " and blue" in result
