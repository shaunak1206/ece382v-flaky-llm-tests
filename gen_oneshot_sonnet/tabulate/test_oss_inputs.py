"""Tests for the input types and header handling of tabulate()."""

from collections import namedtuple
from dataclasses import dataclass

import pytest

from tabulate import tabulate


def lines(*ls):
    return "\n".join(ls)


@dataclass
class Item:
    name: str
    qty: int


Point = namedtuple("Point", ["x", "y"])


def test_list_of_lists_without_headers():
    expected = lines(
        "---  ---------",
        "  1      2.34",
        "-56      8.999",
        "  2  10001",
        "---  ---------",
    )
    assert tabulate([[1, 2.34], [-56, "8.999"], ["2", "10001"]]) == expected


def test_headers_firstrow_pads_short_header_row():
    data = [["sex", "age"], ["Alice", "F", 24], ["Bob", "M", 19]]
    expected = lines(
        "       sex      age",
        "-----  -----  -----",
        "Alice  F         24",
        "Bob    M         19",
    )
    assert tabulate(data, headers="firstrow") == expected


def test_headers_keys_for_list_of_lists_uses_column_indices():
    expected = lines(
        "  0    1",
        "---  ---",
        "  1    2",
        "  3    4",
    )
    assert tabulate([[1, 2], [3, 4]], headers="keys") == expected


def test_dict_of_iterables_with_keys_headers():
    expected = lines(
        "  a    b",
        "---  ---",
        "  1    3",
        "  2    4",
    )
    assert tabulate({"a": [1, 2], "b": [3, 4]}, headers="keys") == expected
    # shorter columns are padded with missing values
    ragged = lines(
        "  a    b",
        "---  ---",
        "  1    4",
        "  2",
        "  3",
    )
    assert tabulate({"a": [1, 2, 3], "b": [4]}, headers="keys") == ragged


def test_list_of_dicts_with_keys_headers():
    data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    expected = lines(
        "  x    y",
        "---  ---",
        "  1    2",
        "  3    4",
    )
    assert tabulate(data, headers="keys") == expected


def test_list_of_dicts_uses_union_of_keys_in_first_seen_order():
    data = [{"a": 1}, {"b": 2}]
    expected = "\n".join(
        [
            "  a    b",
            "---  ---",
            "  1",
            " " * 7 + "2",
        ]
    )
    assert tabulate(data, headers="keys") == expected


def test_list_of_dicts_with_a_dict_of_header_names():
    data = [{"a": 1, "b": 2}]
    expected = "\n".join(
        [
            "  Alpha    Beta",
            "-------  ------",
            "      1" + "  " + "     2",
        ]
    )
    assert tabulate(data, headers={"a": "Alpha", "b": "Beta"}) == expected


def test_list_of_dicts_rejects_a_list_of_headers():
    with pytest.raises(ValueError):
        tabulate([{"a": 1}], headers=["A"])


def test_list_of_namedtuples_with_keys_headers():
    expected = lines(
        "  x    y",
        "---  ---",
        "  1    2",
        "  3    4",
    )
    assert tabulate([Point(1, 2), Point(3, 4)], headers="keys") == expected


def test_list_of_dataclasses():
    items = [Item("apple", 3), Item("kiwi", 12)]
    with_keys = lines(
        "name" + " " * 6 + "qty",
        "------  -----",
        "apple" + " " * 7 + "3",
        "kiwi" + " " * 7 + "12",
    )
    assert tabulate(items, headers="keys") == with_keys
    without_headers = lines(
        "--  --",
        "a    1",
        "bb  22",
        "--  --",
    )
    assert tabulate([Item("a", 1), Item("bb", 22)]) == without_headers


def test_iterators_and_tuples_are_accepted():
    expected = lines("-  -", "1  2", "3  4", "-  -")
    assert tabulate(iter([[1, 2], [3, 4]])) == expected
    assert tabulate(((1, 2), (3, 4))) == expected
    assert tabulate([(1, 2), (3, 4)]) == expected


def test_empty_inputs():
    assert tabulate([]) == ""
    assert tabulate(None) == ""
    assert tabulate([], headers=["a", "b"]) == lines("a    b", "---  ---")


def test_non_iterable_input_raises_type_error():
    with pytest.raises(TypeError):
        tabulate(5)
    with pytest.raises(TypeError):
        tabulate({"a": 1})


def test_headers_shorter_than_columns_label_the_last_columns():
    expected = lines(
        " " * 6 + "b" + " " * 4 + "c",
        "--  ---  ---",
        " 1" + " " * 4 + "2" + " " * 4 + "3",
    )
    assert tabulate([[1, 2, 3]], headers=["b", "c"]) == expected


def test_showindex_always_and_custom_index():
    always = lines(
        "-  -  --",
        "0  F  24",
        "1  M  19",
        "-  -  --",
    )
    data = [["F", 24], ["M", 19]]
    assert tabulate(data, showindex="always") == always
    assert tabulate(data, showindex=True) == always
    custom = lines(
        "-  -  --",
        "a  F  24",
        "b  M  19",
        "-  -  --",
    )
    assert tabulate(data, showindex=["a", "b"]) == custom
    assert tabulate(data, showindex=iter(["a", "b"])) == custom


def test_showindex_with_headers_adds_an_unnamed_first_column():
    expected = lines(
        "    sex      age",
        "--  -----  -----",
        " 0  F" + " " * 9 + "24",
        " 1  M" + " " * 9 + "19",
    )
    out = tabulate([["F", 24], ["M", 19]], headers=["sex", "age"], showindex="always")
    assert out == expected


def test_showindex_never_and_length_mismatch():
    data = [["F", 24], ["M", 19]]
    assert tabulate(data, showindex="never") == tabulate(data)
    assert tabulate(data, showindex=False) == tabulate(data)
    with pytest.raises(ValueError):
        tabulate(data, showindex=[1, 2, 3])


def test_bytes_and_bool_cells():
    assert tabulate([[b"abc", 1]]) == lines("---  -", "abc  1", "---  -")
    assert tabulate([[True, False]]) == lines("----  -----", "True  False", "----  -----")
