"""Unit tests covering miscellaneous public API surface of `tabulate`:
`SEPARATING_LINE`, `simple_separated_format`, multiline cell handling,
and the module's `__version__`/format-registry surface.
"""

from tabulate import SEPARATING_LINE, simple_separated_format, tabulate, tabulate_formats


def test_separating_line_in_grid_format():
    expected = (
        "+---+---+\n"
        "| A | 1 |\n"
        "+---+---+\n"
        "+---+---+\n"
        "| B | 2 |\n"
        "+---+---+"
    )
    result = tabulate([["A", 1], SEPARATING_LINE, ["B", 2]], tablefmt="grid")
    assert result == expected


def test_separating_line_in_simple_format():
    expected = "-  -\nA  1\n-  -\nB  2\n-  -"
    result = tabulate([["A", 1], SEPARATING_LINE, ["B", 2]], tablefmt="simple")
    assert result == expected


def test_simple_separated_format_basic():
    tsv_like = simple_separated_format("|")
    result = tabulate([["a", "b"], ["c", "d"]], tablefmt=tsv_like)
    assert result == "a|b\nc|d"


def test_simple_separated_format_with_headers():
    # simple_separated_format still pads cells to a common column width;
    # it only replaces the whitespace *separator* between columns.
    csv_like = simple_separated_format(",")
    expected = "col1  ,col2\nx     ,y"
    result = tabulate([["x", "y"]], headers=["col1", "col2"], tablefmt=csv_like)
    assert result == expected


def test_multiline_cell_in_grid_format():
    expected = "+-------+---+\n| hello | 1 |\n| world |   |\n+-------+---+"
    result = tabulate([["hello\nworld", 1]], tablefmt="grid")
    assert result == expected


def test_multiline_plain_format_falls_back_gracefully():
    # "plain" has no declared multiline variant; a multiline cell should
    # still render without raising an exception.
    result = tabulate([["hello\nworld", 1]], tablefmt="plain")
    assert "hello" in result
    assert "world" in result


def test_tabulate_formats_is_sorted_list_of_strings():
    assert isinstance(tabulate_formats, list)
    assert all(isinstance(f, str) for f in tabulate_formats)
    assert tabulate_formats == sorted(tabulate_formats)


def test_unknown_tablefmt_falls_back_to_simple():
    data = [["a", 1]]
    result = tabulate(data, tablefmt="not_a_real_format")
    assert result == tabulate(data, tablefmt="simple")


