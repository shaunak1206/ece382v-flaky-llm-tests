"""Tests for number/string formatting and alignment options of tabulate."""

import datetime
from decimal import Decimal

import pytest

from tabulate import tabulate


def lines(*parts):
    return "\n".join(parts)


def test_floatfmt_string_applies_to_all_float_columns():
    out = tabulate([[1.23456], [2.5]], floatfmt=".2f")
    assert out == lines("----", "1.23", "2.50", "----")


def test_floatfmt_list_gives_per_column_formats():
    out = tabulate([[1.5, 2.5]], floatfmt=(".1f", ".3f"))
    assert out == lines("---  -----", "1.5  2.500", "---  -----")


def test_intfmt_applies_to_integer_columns():
    out = tabulate([[1000000]], intfmt=",")
    assert out == lines("---------", "1,000,000", "---------")


def test_default_float_format_uses_general_notation():
    assert tabulate([[0.1 + 0.2]]) == lines("---", "0.3", "---")
    assert "1.23457e+06" in tabulate([[1234567.0]])


def test_decimal_alignment_lines_up_decimal_points():
    out = tabulate([[1.5], [22.25], [333]])
    assert out == lines("------", "  1.5", " 22.25", "333", "------")


def test_numalign_left_and_right():
    assert tabulate([[1], [22]], numalign="left") == lines("--", "1", "22", "--")
    assert tabulate([[1], [22]], numalign="right") == lines("--", " 1", "22", "--")


def test_numalign_center():
    out = tabulate([[1], [22222]], numalign="center")
    assert out == lines("-----", "  1", "22222", "-----")


def test_stralign_right_and_center():
    assert tabulate([["a"], ["ccc"]], stralign="right") == lines("---", "  a", "ccc", "---")
    assert tabulate([["aa"], ["cccc"]], stralign="center") == lines(
        "----", " aa", "cccc", "----"
    )


def test_colalign_overrides_individual_columns():
    out = tabulate([["a", 1], ["bb", 22]], colalign=("right", "left"))
    assert out == lines("--  --", " a  1", "bb  22", "--  --")


def test_colglobalalign_applies_to_every_column():
    out = tabulate([["a", 1], ["bbb", 22]], colglobalalign="right")
    assert out == lines("---  --", "  a   1", "bbb  22", "---  --")


def test_string_colalign_emits_warning():
    with pytest.warns(UserWarning, match="colalign"):
        tabulate([["a", "b"]], colalign="right")


def test_headersalign_overrides_header_alignment():
    out = tabulate([["a", 1]], headers=["h1", "h2"], headersalign=("right", "left"))
    assert out == lines("  h1  h2", "----  ----", "a        1")


def test_headersglobalalign_center():
    out = tabulate([["ab"]], headers=["name"], headersglobalalign="center")
    assert out == lines(" name", "------", "ab")


def test_string_headersalign_emits_warning():
    with pytest.warns(UserWarning, match="headersalign"):
        tabulate([["a", "b"]], headers=["x", "y"], headersalign="right")


def test_missingval_replaces_none():
    out = tabulate([["a", None]], missingval="N/A")
    assert out == lines("-  ---", "a  N/A", "-  ---")


def test_missingval_list_is_per_column():
    out = tabulate([[None, None]], missingval=["x", "y"])
    assert out == lines("-  -", "x  y", "-  -")


def test_bool_values_render_as_words():
    out = tabulate([[True], [False]])
    assert out == lines("-----", "True", "False", "-----")


def test_numeric_strings_are_right_aligned_unless_numparse_disabled():
    assert tabulate([["1"], ["22"]]) == lines("--", " 1", "22", "--")
    assert tabulate([["1"], ["22"]], disable_numparse=True) == lines("--", "1", "22", "--")


def test_disable_numparse_accepts_column_indices():
    data = [["1", "2"], ["22", "33"]]
    assert tabulate(data) == lines("--  --", " 1   2", "22  33", "--  --")
    assert tabulate(data, disable_numparse=[0]) == lines("--  --", "1    2", "22  33", "--  --")


def test_preserve_whitespace_keeps_leading_spaces():
    data = [["  a"], ["b"]]
    assert tabulate(data) == lines("-", "a", "b", "-")
    assert tabulate(data, preserve_whitespace=True) == lines("---", "  a", "b", "---")


def test_decimal_and_date_values_are_formatted():
    out = tabulate([[Decimal("1.5")], [Decimal("22.25")]])
    assert out == lines("-----", " 1.5", "22.25", "-----")
    out = tabulate([[datetime.date(2020, 1, 2)]])
    assert out == lines("----------", "2020-01-02", "----------")
    assert tabulate([[b"abc"]]) == lines("---", "abc", "---")
