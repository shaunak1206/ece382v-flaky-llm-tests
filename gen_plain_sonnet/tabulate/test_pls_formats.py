"""Tests for the different table output formats of tabulate."""

import pytest

from tabulate import simple_separated_format, tabulate, tabulate_formats

DATA = [["spam", 41.9999], ["eggs", "451.0"]]
HEADERS = ["strings", "numbers"]


def lines(*parts):
    return "\n".join(parts)


def test_default_format_is_simple_without_headers():
    out = tabulate([[1, 2.34], [-56, "8.999"], ["2", "10001"]])
    expected = lines(
        "---  ---------",
        "  1      2.34",
        "-56      8.999",
        "  2  10001",
        "---  ---------",
    )
    assert out == expected


def test_simple_format_with_headers_hides_outer_lines():
    out = tabulate([["a", 1], ["bb", 22]], headers=["name", "n"])
    expected = lines(
        "name      n",
        "------  ---",
        "a         1",
        "bb       22",
    )
    assert out == expected


def test_plain_format_with_headers():
    out = tabulate(DATA, HEADERS, "plain")
    expected = lines(
        "strings      numbers",
        "spam         41.9999",
        "eggs        451",
    )
    assert out == expected


def test_plain_format_without_headers():
    out = tabulate(DATA, tablefmt="plain")
    assert out == lines("spam   41.9999", "eggs  451")


def test_grid_format_with_headers():
    out = tabulate(DATA, HEADERS, "grid")
    expected = lines(
        "+-----------+-----------+",
        "| strings   |   numbers |",
        "+===========+===========+",
        "| spam      |   41.9999 |",
        "+-----------+-----------+",
        "| eggs      |  451      |",
        "+-----------+-----------+",
    )
    assert out == expected


def test_grid_format_without_headers():
    out = tabulate(DATA, tablefmt="grid")
    expected = lines(
        "+------+----------+",
        "| spam |  41.9999 |",
        "+------+----------+",
        "| eggs | 451      |",
        "+------+----------+",
    )
    assert out == expected


def test_pipe_format_with_headers_has_alignment_colons():
    out = tabulate(DATA, HEADERS, "pipe")
    expected = lines(
        "| strings   |   numbers |",
        "|:----------|----------:|",
        "| spam      |   41.9999 |",
        "| eggs      |  451      |",
    )
    assert out == expected


def test_pipe_format_without_headers():
    out = tabulate(DATA, tablefmt="pipe")
    expected = lines(
        "|:-----|---------:|",
        "| spam |  41.9999 |",
        "| eggs | 451      |",
    )
    assert out == expected


def test_github_format_is_alias_of_pipe_and_pipe_escapes_bars():
    assert tabulate(DATA, HEADERS, "github") == tabulate(DATA, HEADERS, "pipe")
    out = tabulate([["a|b"]], tablefmt="pipe")
    assert "a\\|b" in out


def test_rst_format_with_headers():
    out = tabulate(DATA, HEADERS, "rst")
    expected = lines(
        "=========  =========",
        "strings      numbers",
        "=========  =========",
        "spam         41.9999",
        "eggs        451",
        "=========  =========",
    )
    assert out == expected


def test_rst_format_escapes_empty_first_column():
    out = tabulate([["", "a"]], tablefmt="rst")
    assert out == lines("==  =", "..  a", "==  =")


def test_psql_format_with_headers():
    out = tabulate([["a", 1]], ["x", "y"], "psql")
    expected = lines(
        "+-----+-----+",
        "| x   |   y |",
        "|-----+-----|",
        "| a   |   1 |",
        "+-----+-----+",
    )
    assert out == expected


def test_pretty_format_centers_and_treats_numbers_as_text():
    out = tabulate([["a", 1], ["bb", 22]], ["x", "y"], "pretty")
    expected = lines(
        "+----+----+",
        "| x  | y  |",
        "+----+----+",
        "| a  | 1  |",
        "| bb | 22 |",
        "+----+----+",
    )
    assert out == expected


def test_simple_grid_uses_box_drawing_characters():
    out = tabulate(DATA, HEADERS, "simple_grid")
    expected = lines(
        "┌───────────┬───────────┐",
        "│ strings   │   numbers │",
        "├───────────┼───────────┤",
        "│ spam      │   41.9999 │",
        "├───────────┼───────────┤",
        "│ eggs      │  451      │",
        "└───────────┴───────────┘",
    )
    assert out == expected


def test_double_outline_has_no_lines_between_rows():
    out = tabulate(DATA, HEADERS, "double_outline")
    expected = lines(
        "╔═══════════╦═══════════╗",
        "║ strings   ║   numbers ║",
        "╠═══════════╬═══════════╣",
        "║ spam      ║   41.9999 ║",
        "║ eggs      ║  451      ║",
        "╚═══════════╩═══════════╝",
    )
    assert out == expected


def test_mediawiki_format():
    data = [["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]]
    out = tabulate(data, headers="firstrow", tablefmt="mediawiki")
    expected = lines(
        '{| class="wikitable" style="text-align: left;"',
        "|+ <!-- caption -->",
        "|-",
        '! strings   !! style="text-align: right;"|   numbers',
        "|-",
        '| spam      || style="text-align: right;"|   41.9999',
        "|-",
        '| eggs      || style="text-align: right;"|  451',
        "|}",
    )
    assert out == expected


def test_html_format_with_headers():
    data = [["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]]
    out = tabulate(data, headers="firstrow", tablefmt="html")
    expected = lines(
        "<table>",
        "<thead>",
        '<tr><th style="text-align: left;">strings  </th>'
        '<th style="text-align: right;">  numbers</th></tr>',
        "</thead>",
        "<tbody>",
        '<tr><td style="text-align: left;">spam     </td>'
        '<td style="text-align: right;">  41.9999</td></tr>',
        '<tr><td style="text-align: left;">eggs     </td>'
        '<td style="text-align: right;"> 451     </td></tr>',
        "</tbody>",
        "</table>",
    )
    assert out == expected


def test_html_escapes_cells_but_unsafehtml_does_not():
    safe = tabulate([["<b>x</b>"]], tablefmt="html")
    unsafe = tabulate([["<b>x</b>"]], tablefmt="unsafehtml")
    assert "&lt;b&gt;x&lt;/b&gt;" in safe
    assert "<b>x</b>" not in safe
    assert "<b>x</b>" in unsafe


def test_html_output_is_a_str_usable_in_jupyter():
    out = tabulate([[1, 2]], tablefmt="html")
    assert isinstance(out, str)
    assert out._repr_html_() == out
    assert out.str == out
    assert out.startswith("<table>")


def test_latex_format():
    out = tabulate(DATA, tablefmt="latex")
    expected = lines(
        r"\begin{tabular}{lr}",
        r"\hline",
        r" spam &  41.9999 \\",
        r" eggs & 451      \\",
        r"\hline",
        r"\end{tabular}",
    )
    assert out == expected


def test_latex_escapes_special_characters_but_latex_raw_does_not():
    escaped = tabulate([["a_b & c%"]], tablefmt="latex")
    raw = tabulate([["a_b & c%"]], tablefmt="latex_raw")
    assert r"a\_b \& c\%" in escaped
    assert "a_b & c%" in raw


def test_latex_booktabs_and_longtable_wrappers():
    booktabs = tabulate(DATA, tablefmt="latex_booktabs")
    assert booktabs.splitlines()[0] == r"\begin{tabular}{lr}"
    assert booktabs.splitlines()[1] == r"\toprule"
    assert booktabs.splitlines()[-2] == r"\bottomrule"
    longtable = tabulate(DATA, tablefmt="latex_longtable")
    assert longtable.splitlines()[0] == r"\begin{longtable}{lr}"
    assert longtable.splitlines()[-1] == r"\end{longtable}"


def test_colon_grid_format_forces_left_text_alignment():
    out = tabulate(DATA, HEADERS, "colon_grid")
    expected = lines(
        "+-----------+-----------+",
        "| strings   | numbers   |",
        "+:==========+:==========+",
        "| spam      | 41.9999   |",
        "+-----------+-----------+",
        "| eggs      | 451       |",
        "+-----------+-----------+",
    )
    assert out == expected


@pytest.mark.parametrize("fmt", tabulate_formats)
def test_every_listed_format_renders_a_table_with_headers(fmt):
    out = tabulate([["a", 1], ["b", 2]], headers=["x", "y"], tablefmt=fmt)
    assert isinstance(out, str)
    assert out
    assert "x" in out
    assert "a" in out


def test_unknown_format_falls_back_to_simple():
    assert tabulate(DATA, HEADERS, tablefmt="no-such-format") == tabulate(DATA, HEADERS)


def test_simple_separated_format_uses_given_separator():
    tsv_like = simple_separated_format("\t")
    assert tabulate([["foo", 1], ["spam", 23]], tablefmt=tsv_like) == "foo \t 1\nspam\t23"
    assert tabulate([["a", 1]], tablefmt=simple_separated_format("|")) == "a|1"


def test_tabulate_formats_is_sorted_and_contains_common_names():
    assert tabulate_formats == sorted(tabulate_formats)
    for name in ("simple", "plain", "grid", "pipe", "html", "latex", "tsv", "rst"):
        assert name in tabulate_formats
