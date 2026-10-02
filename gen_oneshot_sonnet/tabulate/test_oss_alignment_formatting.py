"""Tests for alignment, number formatting, wrapping and special cells in tabulate()."""

import pytest

from tabulate import SEPARATING_LINE, tabulate

FRUIT = [["apple", 1], ["kiwi", 100]]
FRUIT_HEADERS = ["fruit", "n"]


def lines(*ls):
    return "\n".join(ls)


def test_default_alignment_is_left_for_text_and_right_for_numbers():
    expected = lines(
        "fruit" + " " * 6 + "n",
        "-------  ---",
        "apple" + " " * 6 + "1",
        "kiwi" + " " * 5 + "100",
    )
    assert tabulate(FRUIT, FRUIT_HEADERS) == expected


NUMALIGN_EXPECTED = {
    "left": lines(
        "fruit" + " " * 4 + "n",
        "-------  ---",
        "apple" + " " * 4 + "1",
        "kiwi" + " " * 5 + "100",
    ),
    "center": lines(
        "fruit" + " " * 5 + "n",
        "-------  ---",
        "apple" + " " * 5 + "1",
        "kiwi" + " " * 5 + "100",
    ),
    "right": lines(
        "fruit" + " " * 6 + "n",
        "-------  ---",
        "apple" + " " * 6 + "1",
        "kiwi" + " " * 5 + "100",
    ),
}


@pytest.mark.parametrize("numalign", sorted(NUMALIGN_EXPECTED))
def test_numalign(numalign):
    out = tabulate(FRUIT, FRUIT_HEADERS, numalign=numalign)
    assert out == NUMALIGN_EXPECTED[numalign]


STRALIGN_EXPECTED = {
    "right": lines(
        "  fruit    n",
        "-------  ---",
        "  apple    1",
        "   kiwi  100",
    ),
    "center": lines(
        " fruit" + " " * 5 + "n",
        "-------  ---",
        " apple" + " " * 5 + "1",
        " kiwi" + " " * 4 + "100",
    ),
}


@pytest.mark.parametrize("stralign", sorted(STRALIGN_EXPECTED))
def test_stralign(stralign):
    out = tabulate(FRUIT, FRUIT_HEADERS, stralign=stralign)
    assert out == STRALIGN_EXPECTED[stralign]


def test_stralign_none_disables_padding_of_text_columns():
    out = tabulate([["a", "bbb"], ["cc", "d"]], stralign=None)
    assert out == lines("--  ---", "a  bbb", "cc  d", "--  ---")


def test_colalign_colglobalalign_and_global_keyword():
    per_column = lines(
        "  fruit  n",
        "-------  ---",
        "  apple  1",
        "   kiwi  100",
    )
    assert tabulate(FRUIT, FRUIT_HEADERS, colalign=("right", "left")) == per_column
    # colglobalalign applies to every column before colalign overrides
    all_right = lines(
        "  fruit    n",
        "-------  ---",
        "  apple    1",
        "   kiwi  100",
    )
    assert tabulate(FRUIT, FRUIT_HEADERS, colglobalalign="right") == all_right
    assert tabulate(FRUIT, FRUIT_HEADERS, colglobalalign="right", colalign=["global", "right"]) == all_right
    # "global" leaves the default in place for that column
    assert tabulate(FRUIT, FRUIT_HEADERS, colalign=["global", "left"]) == NUMALIGN_EXPECTED["left"]


def test_colalign_given_as_a_string_warns():
    with pytest.warns(UserWarning):
        tabulate(FRUIT, FRUIT_HEADERS, colalign="right")


def test_header_alignment_overrides():
    centered = lines(
        " fruit" + " " * 4 + "n",
        "-------  ---",
        "apple" + " " * 6 + "1",
        "kiwi" + " " * 5 + "100",
    )
    assert tabulate(FRUIT, FRUIT_HEADERS, headersglobalalign="center") == centered
    right_first = lines(
        "  fruit    n",
        "-------  ---",
        "apple" + " " * 6 + "1",
        "kiwi" + " " * 5 + "100",
    )
    assert tabulate(FRUIT, FRUIT_HEADERS, headersalign=["right"]) == right_first


def test_floatfmt_string_and_list():
    data = [[1.23456, 2.5]]
    assert tabulate(data, floatfmt=".2f") == lines("----  ----", "1.23  2.50", "----  ----")
    assert tabulate(data, floatfmt=[".1f", ".3f"]) == lines("---  -----", "1.2  2.500", "---  -----")
    # an int mixed into a float column is rendered with the float format too
    assert tabulate([[1.5], [2]], floatfmt=".1f") == lines("---", "1.5", "2.0", "---")


def test_intfmt_string_and_list():
    data = [[1234567, 5]]
    assert tabulate(data, intfmt=",") == lines("---------  -", "1,234,567  5", "---------  -")
    assert tabulate(data, intfmt=[",", "04d"]) == lines(
        "---------  ----", "1,234,567  0005", "---------  ----"
    )


def test_missingval_string_default_and_per_column():
    data = [["a", None], ["b", 2]]
    assert tabulate(data, missingval="?") == lines("-  -", "a  ?", "b  2", "-  -")
    assert tabulate(data) == lines("-  -", "a", "b  2", "-  -")
    assert tabulate([["x", None, None]], missingval=["", "A", "B"]) == lines(
        "-  -  -", "x  A  B", "-  -  -"
    )


def test_disable_numparse_variants():
    data = [["1", "2.50"], ["10", "3"]]
    assert tabulate(data) == lines("--  ---", " 1  2.5", "10  3", "--  ---")
    assert tabulate(data, disable_numparse=True) == lines("--  ----", "1   2.50", "10  3", "--  ----")
    assert tabulate(data, disable_numparse=[0]) == lines("--  ---", "1   2.5", "10  3", "--  ---")


def test_numeric_looking_strings_are_parsed_unless_disabled():
    # "42992e1" is a valid float literal, so it gets reformatted
    assert tabulate([["42992e1"]]) == lines("------", "429920", "------")
    assert tabulate([["42992e1"]], disable_numparse=True) == lines(
        "-------", "42992e1", "-------"
    )
    # strings with thousands separators are numbers and align to the right
    assert tabulate([["1,000"], ["20"]]) == lines("-----", "1,000", "   20", "-----")


def test_preserve_whitespace():
    data = [["  a", "b  "]]
    assert tabulate(data) == lines("-  -", "a  b", "-  -")
    assert tabulate(data, preserve_whitespace=True) == lines("---  ---", "  a  b", "---  ---")


def test_multiline_cells_in_simple_and_grid_formats():
    assert tabulate([["a\nb", 1]]) == lines("-  -", "a  1", "b", "-  -")
    assert tabulate([["a\nb", "c"]], tablefmt="grid") == lines(
        "+---+---+",
        "| a | c |",
        "| b |   |",
        "+---+---+",
    )
    with_headers = tabulate([["a\nb", "c"]], headers=["h1", "h2"], tablefmt="grid")
    assert with_headers == lines(
        "+------+------+",
        "| h1   | h2   |",
        "+======+======+",
        "| a    | c    |",
        "| b    |      |",
        "+------+------+",
    )
    bottom = tabulate([["a\nb", "c"]], tablefmt="grid", rowalign="bottom")
    assert bottom == lines(
        "+---+---+",
        "| a |   |",
        "| b | c |",
        "+---+---+",
    )


def test_separating_line_between_rows():
    rows = [["a", 1], SEPARATING_LINE, ["b", 2]]
    assert tabulate(rows) == lines("-  -", "a  1", "-  -", "b  2", "-  -")
    assert tabulate(rows, tablefmt="psql") == lines(
        "+---+---+",
        "| a | 1 |",
        "|---+---|",
        "| b | 2 |",
        "+---+---+",
    )
    assert tabulate(rows, tablefmt="plain") == "a  1\n\nb  2"


def test_maxcolwidths_wraps_long_cells():
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
    assert out == lines(
        "+------------+------------+-------------------------------+",
        "|   Issue Id | Author     | Description                   |",
        "+============+============+===============================+",
        "|          1 | John Smith | This is a rather long         |",
        "|            |            | description that might look   |",
        "|            |            | better if it is wrapped a bit |",
        "+------------+------------+-------------------------------+",
    )
    # a single int applies to every column
    assert tabulate([["abcdef ghijkl"]], maxcolwidths=6) == lines(
        "------", "abcdef", "ghijkl", "------"
    )


def test_maxheadercolwidths_wraps_long_headers():
    out = tabulate([[1]], headers=["long header"], maxheadercolwidths=[6])
    assert out == lines("    long", "  header", "--------", "       1")


def test_ansi_color_codes_do_not_count_towards_width():
    red = "\x1b[31mred\x1b[0m"
    assert tabulate([[red, 1]]) == lines("---  -", red + "  1", "---  -")
    out = tabulate([[red, 1], ["green", 2]])
    assert out == lines("-----  -", red + "    1", "green  2", "-----  -")


def test_unicode_text_is_aligned_by_character_count():
    headers = ["буква", "цифра"]
    rows = [["аз", 2], ["буки", 4]]
    expected = lines(
        headers[0] + " " * 6 + headers[1],
        "-------  -------",
        rows[0][0] + " " * 13 + "2",
        rows[1][0] + " " * 11 + "4",
    )
    assert tabulate(rows, headers) == expected
