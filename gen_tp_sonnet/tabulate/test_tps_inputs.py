import copy
from collections import namedtuple
from dataclasses import dataclass

import tabulate

T = tabulate.tabulate


def _rows(out):
    return [line.split() for line in out.splitlines()]


def test_list_of_lists():
    assert _rows(T([[1, 2], [3, 4]], tablefmt="plain")) == [["1", "2"], ["3", "4"]]


def test_tuple_of_tuples():
    assert _rows(T(((1, "a"), (2, "b")), tablefmt="plain")) == [["1", "a"], ["2", "b"]]


def test_generator_of_rows():
    gen = ((i, i * i) for i in range(3))
    assert _rows(T(gen, tablefmt="plain")) == [["0", "0"], ["1", "1"], ["2", "4"]]


def test_explicit_headers_list():
    out = T([[1, 2]], headers=["a", "b"], tablefmt="plain")
    assert _rows(out) == [["a", "b"], ["1", "2"]]


def test_headers_firstrow():
    out = T([["name", "age"], ["Al", 3]], headers="firstrow", tablefmt="plain")
    assert _rows(out) == [["name", "age"], ["Al", "3"]]


def test_dict_of_iterables_with_keys_headers():
    out = T({"a": [1, 2], "b": [3, 4]}, headers="keys", tablefmt="plain")
    assert _rows(out) == [["a", "b"], ["1", "3"], ["2", "4"]]


def test_list_of_dicts_with_keys_headers():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    out = T(data, headers="keys", tablefmt="plain")
    assert _rows(out) == [["a", "b"], ["1", "2"], ["3", "4"]]


def test_namedtuple_rows_with_keys_headers():
    Point = namedtuple("Point", ["x", "y"])
    out = T([Point(1, 2), Point(3, 4)], headers="keys", tablefmt="plain")
    assert _rows(out) == [["x", "y"], ["1", "2"], ["3", "4"]]


def test_dataclass_rows_with_keys_headers():
    @dataclass
    class Item:
        name: str
        qty: int

    out = T([Item("spam", 3), Item("eggs", 5)], headers="keys", tablefmt="plain")
    assert _rows(out) == [["name", "qty"], ["spam", "3"], ["eggs", "5"]]


def test_fewer_headers_than_columns():
    out = T([[1, 2, 3]], headers=["a", "b"], tablefmt="plain")
    lines = out.splitlines()
    assert lines[0].split() == ["a", "b"]
    assert lines[1].split() == ["1", "2", "3"]


def test_showindex_true():
    assert T([["a"], ["b"]], showindex=True, tablefmt="plain") == "0  a\n1  b"


def test_showindex_always():
    assert T([["a"], ["b"]], showindex="always", tablefmt="plain") == "0  a\n1  b"


def test_showindex_never_and_false_hide_index():
    data = [["a"], ["b"]]
    assert T(data, showindex="never", tablefmt="plain") == "a\nb"
    assert T(data, showindex=False, tablefmt="plain") == "a\nb"


def test_showindex_custom_iterable():
    out = T([["a"], ["b"]], showindex=["x", "y"], tablefmt="plain")
    assert out == "x  a\ny  b"


def test_empty_data_returns_empty_string():
    assert T([]) == ""


def test_ragged_rows_are_padded():
    lines = T([[1, 2], [3]], tablefmt="plain").splitlines()
    assert lines[0].split() == ["1", "2"]
    assert lines[1].split() == ["3"]


def test_input_is_not_mutated():
    data = [[1, "a"], [2, "b"]]
    snapshot = copy.deepcopy(data)
    T(data, headers=["n", "s"], tablefmt="grid")
    assert data == snapshot


def test_none_cell_defaults_to_empty():
    lines = T([["a", None], ["b", "c"]], tablefmt="plain").splitlines()
    assert lines[0].strip() == "a"
    assert lines[1].split() == ["b", "c"]
