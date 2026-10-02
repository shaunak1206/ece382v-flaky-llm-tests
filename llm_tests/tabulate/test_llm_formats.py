"""Unit tests covering the major `tablefmt` values supported by `tabulate()`.

These tests use a fixed, small data set (mix of strings and floats) so the
exact rendered output for each table format can be verified byte-for-byte.
"""

import pytest

from tabulate import tabulate

DATA = [["spam", 41.9999], ["eggs", "451.0"]]
HEADERS = ["strings", "numbers"]


def test_plain_format():
    expected = "strings      numbers\nspam         41.9999\neggs        451"
    assert tabulate(DATA, HEADERS, tablefmt="plain") == expected


def test_simple_format():
    expected = (
        "strings      numbers\n"
        "---------  ---------\n"
        "spam         41.9999\n"
        "eggs        451"
    )
    assert tabulate(DATA, HEADERS, tablefmt="simple") == expected


def test_grid_format():
    expected = (
        "+-----------+-----------+\n"
        "| strings   |   numbers |\n"
        "+===========+===========+\n"
        "| spam      |   41.9999 |\n"
        "+-----------+-----------+\n"
        "| eggs      |  451      |\n"
        "+-----------+-----------+"
    )
    assert tabulate(DATA, HEADERS, tablefmt="grid") == expected


def test_github_format():
    expected = (
        "| strings   |   numbers |\n"
        "|:----------|----------:|\n"
        "| spam      |   41.9999 |\n"
        "| eggs      |  451      |"
    )
    assert tabulate(DATA, HEADERS, tablefmt="github") == expected


def test_pipe_format():
    expected = (
        "| strings   |   numbers |\n"
        "|:----------|----------:|\n"
        "| spam      |   41.9999 |\n"
        "| eggs      |  451      |"
    )
    assert tabulate(DATA, HEADERS, tablefmt="pipe") == expected


def test_orgtbl_format():
    expected = (
        "| strings   |   numbers |\n"
        "|-----------+-----------|\n"
        "| spam      |   41.9999 |\n"
        "| eggs      |  451      |"
    )
    assert tabulate(DATA, HEADERS, tablefmt="orgtbl") == expected


def test_rst_format():
    expected = (
        "=========  =========\n"
        "strings      numbers\n"
        "=========  =========\n"
        "spam         41.9999\n"
        "eggs        451\n"
        "=========  ========="
    )
    assert tabulate(DATA, HEADERS, tablefmt="rst") == expected


def test_html_format():
    expected = (
        "<table>\n"
        "<thead>\n"
        '<tr><th style="text-align: left;">strings  </th>'
        '<th style="text-align: right;">  numbers</th></tr>\n'
        "</thead>\n"
        "<tbody>\n"
        '<tr><td style="text-align: left;">spam     </td>'
        '<td style="text-align: right;">  41.9999</td></tr>\n'
        '<tr><td style="text-align: left;">eggs     </td>'
        '<td style="text-align: right;"> 451     </td></tr>\n'
        "</tbody>\n"
        "</table>"
    )
    assert tabulate(DATA, HEADERS, tablefmt="html") == expected


def test_unsafehtml_does_not_escape_markup():
    data = [["<b>bold</b>", 1]]
    result = tabulate(data, tablefmt="unsafehtml")
    assert "<b>bold</b>" in result


def test_html_escapes_markup():
    data = [["<b>bold</b>", 1]]
    result = tabulate(data, tablefmt="html")
    assert "<b>bold</b>" not in result
    assert "&lt;b&gt;bold&lt;/b&gt;" in result


def test_latex_format():
    expected = (
        "\\begin{tabular}{lr}\n"
        "\\hline\n"
        " strings   &   numbers \\\\\n"
        "\\hline\n"
        " spam      &   41.9999 \\\\\n"
        " eggs      &  451      \\\\\n"
        "\\hline\n"
        "\\end{tabular}"
    )
    assert tabulate(DATA, HEADERS, tablefmt="latex") == expected


def test_latex_raw_does_not_escape_while_latex_does():
    escaped = tabulate([["a_b", 1]], tablefmt="latex")
    raw = tabulate([["a_b", 1]], tablefmt="latex_raw")
    assert "a\\_b" in escaped
    assert "a_b" in raw and "a\\_b" not in raw


def test_tsv_format():
    result = tabulate(DATA, HEADERS, tablefmt="tsv")
    lines = result.split("\n")
    assert "\t" in lines[0]
    assert lines[0].split("\t")[0].strip() == "strings"


def test_fancy_grid_format():
    result = tabulate(DATA, HEADERS, tablefmt="fancy_grid")
    assert result.startswith("\u2552")
    assert "strings" in result
    assert "\u2500" in result  # thin divider between data rows


def test_psql_format():
    expected = (
        "+-----------+-----------+\n"
        "| strings   |   numbers |\n"
        "|-----------+-----------|\n"
        "| spam      |   41.9999 |\n"
        "| eggs      |  451      |\n"
        "+-----------+-----------+"
    )
    assert tabulate(DATA, HEADERS, tablefmt="psql") == expected


def test_jira_format():
    expected = (
        "|| strings   ||   numbers ||\n"
        "| spam      |   41.9999 |\n"
        "| eggs      |  451      |"
    )
    assert tabulate(DATA, HEADERS, tablefmt="jira") == expected


def test_mediawiki_format():
    result = tabulate(
        [["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]],
        headers="firstrow",
        tablefmt="mediawiki",
    )
    assert result.startswith('{| class="wikitable" style="text-align: left;"')
    assert result.endswith("|}")
    assert "|-" in result


@pytest.mark.parametrize(
    "fmt",
    [
        "plain",
        "simple",
        "grid",
        "github",
        "pipe",
        "orgtbl",
        "rst",
        "html",
        "latex",
        "tsv",
        "fancy_grid",
        "psql",
        "jira",
        "mediawiki",
        "unsafehtml",
    ],
)
def test_all_common_formats_produce_nonempty_string_containing_data(fmt):
    result = tabulate(DATA, HEADERS, tablefmt=fmt)
    assert isinstance(str(result), str)
    assert "spam" in str(result)
    assert "eggs" in str(result)


def test_colon_grid_default_alignment():
    expected = (
        "+-----------+-----------+\n"
        "| strings   | numbers   |\n"
        "+:==========+:==========+\n"
        "| spam      | 41.9999   |\n"
        "+-----------+-----------+\n"
        "| eggs      | 451       |\n"
        "+-----------+-----------+"
    )
    assert tabulate(DATA, HEADERS, tablefmt="colon_grid") == expected


