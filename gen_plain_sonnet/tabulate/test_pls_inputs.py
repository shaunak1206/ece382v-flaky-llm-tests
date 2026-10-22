"""Tests for the supported input data shapes and header handling."""

from collections import namedtuple
from dataclasses import dataclass

import pytest

from tabulate import tabulate


def lines(*parts):
    return "\n".join(parts)


@dataclass
class Point:
    x: int
    y: int


def test_empty_data_gives_empty_string():
    assert tabulate([]) == ""
    assert tabulate(None) == ""


def test_empty_data_with_headers_still_prints_header():
    out = tabulate([], headers=["a", "b"])
    assert out == lines("a    b", "---  ---")


def test_firstrow_headers_with_short_header_row():
    out = tabulate([["sex", "age"], ["Alice", "F", 24], ["Bob", "M", 19]], headers="firstrow")
    expected = lines(
        "       sex      age",
        "-----  -----  -----",
        "Alice  F         24",
        "Bob    M         19",
    )
    assert out == expected


def test_keys_headers_on_list_of_lists_use_column_indices():
    out = tabulate([[1, 2], [3, 4]], headers="keys")
    assert out == lines("  0    1", "---  ---", "  1    2", "  3    4")


def test_list_of_dicts_with_keys_headers():
    out = tabulate([{"a": 1, "b": 2}, {"a": 3, "b": 4}], headers="keys")
    assert out == lines("  a    b", "---  ---", "  1    2", "  3    4")


def test_list_of_dicts_with_dict_headers_renames_columns():
    out = tabulate([{"a": 1}], headers={"a": "Alpha"})
    assert out == lines("  Alpha", "-------", "      1")


def test_list_of_dicts_with_list_headers_raises():
    with pytest.raises(ValueError):
        tabulate([{"a": 1}], headers=["x"])


def test_dict_of_lists_uses_keys_as_columns():
    out = tabulate({"x": [1, 2], "y": [3, 4]}, headers="keys")
    assert out == lines("  x    y", "---  ---", "  1    3", "  2    4")


def test_namedtuple_rows_provide_field_names():
    Pair = namedtuple("Pair", ["x", "y"])
    out = tabulate([Pair(1, 2)], headers="keys")
    assert out == lines("  x    y", "---  ---", "  1    2")


def test_dataclass_rows_provide_field_names():
    out = tabulate([Point(1, 2), Point(3, 4)], headers="keys")
    assert out == lines("  x    y", "---  ---", "  1    2", "  3    4")


def test_non_iterable_data_raises_type_error():
    with pytest.raises(TypeError):
        tabulate(5)
    with pytest.raises(TypeError):
        tabulate({"a": 1})


def test_showindex_always_adds_row_numbers():
    out = tabulate([["F", 24], ["M", 19]], showindex="always")
    assert out == lines("-  -  --", "0  F  24", "1  M  19", "-  -  --")


def test_showindex_true_equals_always_and_never_adds_nothing():
    data = [["a"], ["b"]]
    assert tabulate(data, showindex=True) == lines("-  -", "0  a", "1  b", "-  -")
    assert tabulate(data, showindex="never") == tabulate(data)
    assert tabulate(data, showindex=False) == tabulate(data)


def test_showindex_with_custom_values():
    out = tabulate([["a"], ["b"]], showindex=[10, 20])
    assert out == lines("--  -", "10  a", "20  b", "--  -")


def test_showindex_with_wrong_length_raises():
    with pytest.raises(ValueError, match="index must be as long"):
        tabulate([["a"], ["b"]], showindex=[1])


def test_showindex_with_headers_pads_header_for_index_column():
    out = tabulate([["a"], ["b"]], headers=["x"], showindex="always")
    assert out == lines("    x", "--  ---", " 0  a", " 1  b")


def test_fewer_headers_than_columns_label_the_last_columns():
    out = tabulate([[1, 2, 3]], headers=["b", "c"])
    assert out == lines("      b    c", "--  ---  ---", " 1    2    3")


def test_ragged_rows_are_padded_with_blanks():
    rows = tabulate([[1, 2, 3], [4]]).splitlines()
    assert rows[1].split() == ["1", "2", "3"]
    assert rows[2].strip() == "4"
