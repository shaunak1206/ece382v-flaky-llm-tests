"""Tests for row layout (multiline cells, wrapping, separators) and internal helpers."""

import re
from fractions import Fraction

import tabulate as tabmod
from tabulate import SEPARATING_LINE, tabulate


def lines(*parts):
    return "\n".join(parts)


def test_separating_line_between_rows():
    out = tabulate([["a", 1], SEPARATING_LINE, ["b", 2]])
    assert out == lines("-  -", "a  1", "-  -", "b  2", "-  -")


def test_multiline_cell_in_grid_spans_several_lines():
    out = tabulate([["a\nb", 1]], tablefmt="grid")
    expected = lines(
        "+---+---+",
        "| a | 1 |",
        "| b |   |",
        "+---+---+",
    )
    assert out == expected


def test_rowalign_bottom_and_center_position_short_cells():
    bottom = tabulate([["a\nb", "x"]], tablefmt="grid", rowalign="bottom")
    assert bottom == lines("+---+---+", "| a |   |", "| b | x |", "+---+---+")
    center = tabulate([["a\nb\nc", "x"]], tablefmt="grid", rowalign="center")
    assert center == lines(
        "+---+---+",
        "| a |   |",
        "| b | x |",
        "| c |   |",
        "+---+---+",
    )


def test_maxcolwidths_wraps_long_text():
    out = tabulate([["hello world"]], maxcolwidths=5)
    assert out == lines("-----", "hello", "world", "-----")


def test_maxcolwidths_documented_grid_example():
    out = tabulate(
        [
            (
                "1",
                "John Smith",
                "This is a rather long description that might look better if it is wrapped a bit",
            )
        ],
        headers=("Issue Id", "Author", "Description"),
        maxcolwidths=[None, None, 30],
        tablefmt="grid",
    )
    expected = lines(
        "+------------+------------+-------------------------------+",
        "|   Issue Id | Author     | Description                   |",
        "+============+============+===============================+",
        "|          1 | John Smith | This is a rather long         |",
        "|            |            | description that might look   |",
        "|            |            | better if it is wrapped a bit |",
        "+------------+------------+-------------------------------+",
    )
    assert out == expected


def test_long_words_are_broken_unless_disabled_and_numbers_are_not_wrapped():
    broken = tabulate([["abcdefghij"]], maxcolwidths=4)
    assert broken == lines("----", "abcd", "efgh", "ij", "----")
    intact = tabulate([["abcdefghij"]], maxcolwidths=4, break_long_words=False)
    assert intact == lines("-" * 10, "abcdefghij", "-" * 10)
    number = tabulate([[12345678]], maxcolwidths=3)
    assert "12345678" in number


def test_maxheadercolwidths_wraps_header_text():
    out = tabulate([["x"]], headers=["hello world"], maxheadercolwidths=5)
    rows = out.splitlines()
    assert rows[0] == "hello"
    assert rows[1] == "world"
    assert rows[2] == "-" * 7
    assert rows[3] == "x"


def test_ansi_color_codes_do_not_affect_column_alignment():
    colored = "\x1b[31mred\x1b[0m"
    out = tabulate([[colored, 1], ["blue", 2]])
    plain = [re.sub(r"\x1b\[[0-9;]*m", "", row) for row in out.splitlines()]
    assert plain == ["----  -", "red   1", "blue  2", "----  -"]


def test_number_and_type_detection_helpers():
    assert tabmod._isnumber(123)
    assert tabmod._isnumber("123.45")
    assert tabmod._isnumber("123e45")
    assert tabmod._isnumber("inf")
    assert tabmod._isnumber(Fraction(1, 3))
    assert not tabmod._isnumber("spam")
    assert not tabmod._isnumber("123e45678")
    assert not tabmod._isnumber(None)
    assert tabmod._isint("123")
    assert not tabmod._isint("123.45")
    assert tabmod._isbool("False")
    assert not tabmod._isbool(1)
    assert tabmod._type(None) is type(None)
    assert tabmod._type("") is type(None)
    assert tabmod._type("foo") is str
    assert tabmod._type("1") is int
    assert tabmod._type("1.5") is float
    assert tabmod._type("1,000") is int
    assert tabmod._type("12", numparse=False) is str
    assert tabmod._type(b"abc") is bytes
    assert tabmod._type(True) is bool
    assert tabmod._column_type(["1", "2"]) is int
    assert tabmod._column_type(["1", "2.3"]) is float
    assert tabmod._column_type(["1", "2.3", "four"]) is str
    assert tabmod._column_type([None, "brux"]) is str
    assert tabmod._more_generic(int, float) is float
    assert tabmod._more_generic(type(None), bool) is bool


def test_format_helper_by_value_type():
    assert tabmod._format(1.5, float, "g", "") == "1.5"
    assert tabmod._format(3.14159, float, ".2f", "") == "3.14"
    assert tabmod._format("1,234.5", float, ".1f", "", has_invisible=False) == "1234.5"
    assert tabmod._format(1234567, int, "g", ",") == "1,234,567"
    assert tabmod._format(None, str, "g", "", "X") == "X"
    assert tabmod._format("", float, ".2f", "") == ""
    assert tabmod._format(b"abc", bytes, "g", "") == "abc"
    assert tabmod._format("abc", str, "g", "") == "abc"


def test_padding_afterpoint_and_ansi_helpers():
    assert tabmod._padleft(5, "ab") == "   ab"
    assert tabmod._padright(5, "ab") == "ab   "
    assert tabmod._padboth(6, "ab") == "  ab  "
    assert tabmod._afterpoint("123.45") == 2
    assert tabmod._afterpoint("1001") == -1
    assert tabmod._afterpoint("eggs") == -1
    assert tabmod._afterpoint("123e45") == 2
    assert tabmod._strip_ansi("\x1b[31mred\x1b[0m text") == "red text"
    link = "\x1b]8;;https://example.com\x1b\\This is a link\x1b]8;;\x1b\\"
    assert tabmod._strip_ansi(link) == "This is a link"
