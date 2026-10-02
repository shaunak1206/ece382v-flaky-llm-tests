"""Unit tests covering the different `tabular_data` input types and the
`headers` / `showindex` options of `tabulate()`.
"""

from collections import namedtuple
from dataclasses import dataclass

import pytest

from tabulate import tabulate


def test_list_of_lists_no_headers():
    result = tabulate([[1, 2], [3, 4]])
    expected = "-  -\n1  2\n3  4\n-  -"
    assert result == expected


def test_list_of_tuples():
    result = tabulate([(1, 2), (3, 4)], headers=["A", "B"])
    expected = "  A    B\n---  ---\n  1    2\n  3    4"
    assert result == expected


def test_headers_keys_list_of_dicts():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    expected = "  a    b\n---  ---\n  1    2\n  3    4"
    assert tabulate(data, headers="keys") == expected


def test_list_of_dicts_missing_keys_are_filled():
    data = [{"a": 1, "b": 2}, {"a": 3}]
    result = tabulate(data, headers="keys")
    lines = result.splitlines()
    assert lines[0].split() == ["a", "b"]
    # second row has no "b", should show the empty missing value
    assert "3" in result


def test_headers_keys_dict_of_iterables():
    data = {"a": [1, 3], "b": [2, 4]}
    expected = "  a    b\n---  ---\n  1    2\n  3    4"
    assert tabulate(data, headers="keys") == expected


def test_headers_dict_for_list_of_dicts_renames_columns():
    data = [{"a": 1, "b": 2}]
    result = tabulate(data, headers={"a": "Alpha", "b": "Beta"})
    assert "Alpha" in result
    assert "Beta" in result


def test_headers_keys_namedtuples():
    Point = namedtuple("Point", ["x", "y"])
    pts = [Point(1, 2), Point(3, 4)]
    expected = "  x    y\n---  ---\n  1    2\n  3    4"
    assert tabulate(pts, headers="keys") == expected


def test_headers_keys_dataclasses():
    @dataclass
    class Emp:
        name: str
        age: int

    emps = [Emp("Alice", 30), Emp("Bob", 25)]
    expected = "name      age\n------  -----\nAlice      30\nBob        25"
    assert tabulate(emps, headers="keys") == expected


def test_headers_firstrow():
    data = [["sex", "age"], ["Alice", "F", 24], ["Bob", "M", 19]]
    expected = (
        "       sex      age\n"
        "-----  -----  -----\n"
        "Alice  F         24\n"
        "Bob    M         19"
    )
    assert tabulate(data, headers="firstrow") == expected


def test_explicit_headers_fewer_than_columns():
    # if fewer headers than columns are given, they apply to the
    # right-most columns (like R / pandas)
    data = [["sex", "F", 24], ["sex", "M", 19]]
    result = tabulate(data, headers=["age"])
    header_line = result.splitlines()[0]
    assert header_line.strip() == "age"


def test_showindex_always():
    expected = "-  -  --\n0  F  24\n1  M  19\n-  -  --"
    assert tabulate([["F", 24], ["M", 19]], showindex="always") == expected


def test_showindex_custom_iterable():
    expected = "-  -  --\nx  F  24\ny  M  19\n-  -  --"
    assert tabulate([["F", 24], ["M", 19]], showindex=["x", "y"]) == expected


def test_showindex_never_suppresses_index():
    result = tabulate([["F", 24], ["M", 19]], showindex="never")
    assert result == tabulate([["F", 24], ["M", 19]])


def test_empty_tabular_data_returns_empty_string():
    assert tabulate([]) == ""
    assert tabulate(None) == ""


def test_ragged_rows_are_padded():
    data = [[1, 2, 3], [4, 5]]
    expected = "-  -  -\n1  2  3\n4  5\n-  -  -"
    assert tabulate(data) == expected


@pytest.mark.parametrize("headers_arg", ["keys", []])
def test_dict_of_iterables_headers_variants(headers_arg):
    data = {"a": [1, 3], "b": [2, 4]}
    result = tabulate(data, headers=headers_arg)
    if headers_arg == "keys":
        assert "a" in result.splitlines()[0]
    else:
        assert "a" not in result
