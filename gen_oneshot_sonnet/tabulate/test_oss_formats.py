"""Tests for the tablefmt values supported by tabulate()."""

import pytest

from tabulate import (
    DataRow,
    Line,
    TableFormat,
    simple_separated_format,
    tabulate,
    tabulate_formats,
)

DATA = [["spam", 41.9999], ["eggs", "451.0"]]
HEADERS = ["strings", "numbers"]


def block(text):
    """Strip the leading/trailing newline of a triple-quoted expected block."""
    return text.strip("\n")


def test_simple_with_headers():
    expected = block(
        """
strings      numbers
---------  ---------
spam         41.9999
eggs        451
"""
    )
    assert tabulate(DATA, HEADERS, "simple") == expected
    # "simple" is also the default format
    assert tabulate(DATA, HEADERS) == expected


def test_simple_without_headers():
    expected = block(
        """
----  --------
spam   41.9999
eggs  451
----  --------
"""
    )
    assert tabulate(DATA, tablefmt="simple") == expected


def test_plain_format_with_and_without_headers():
    with_headers = block(
        """
strings      numbers
spam         41.9999
eggs        451
"""
    )
    without_headers = block(
        """
spam   41.9999
eggs  451
"""
    )
    assert tabulate(DATA, HEADERS, "plain") == with_headers
    assert tabulate(DATA, tablefmt="plain") == without_headers


def test_grid_with_headers():
    expected = block(
        """
+-----------+-----------+
| strings   |   numbers |
+===========+===========+
| spam      |   41.9999 |
+-----------+-----------+
| eggs      |  451      |
+-----------+-----------+
"""
    )
    assert tabulate(DATA, HEADERS, "grid") == expected


def test_grid_without_headers():
    expected = block(
        """
+------+----------+
| spam |  41.9999 |
+------+----------+
| eggs | 451      |
+------+----------+
"""
    )
    assert tabulate(DATA, tablefmt="grid") == expected


BOX_GRID_EXPECTED = {
    "simple_grid": """
┌───────────┬───────────┐
│ strings   │   numbers │
├───────────┼───────────┤
│ spam      │   41.9999 │
├───────────┼───────────┤
│ eggs      │  451      │
└───────────┴───────────┘
""",
    "rounded_grid": """
╭───────────┬───────────╮
│ strings   │   numbers │
├───────────┼───────────┤
│ spam      │   41.9999 │
├───────────┼───────────┤
│ eggs      │  451      │
╰───────────┴───────────╯
""",
    "heavy_grid": """
┏━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ strings   ┃   numbers ┃
┣━━━━━━━━━━━╋━━━━━━━━━━━┫
┃ spam      ┃   41.9999 ┃
┣━━━━━━━━━━━╋━━━━━━━━━━━┫
┃ eggs      ┃  451      ┃
┗━━━━━━━━━━━┻━━━━━━━━━━━┛
""",
    "mixed_grid": """
┍━━━━━━━━━━━┯━━━━━━━━━━━┑
│ strings   │   numbers │
┝━━━━━━━━━━━┿━━━━━━━━━━━┥
│ spam      │   41.9999 │
├───────────┼───────────┤
│ eggs      │  451      │
┕━━━━━━━━━━━┷━━━━━━━━━━━┙
""",
    "double_grid": """
╔═══════════╦═══════════╗
║ strings   ║   numbers ║
╠═══════════╬═══════════╣
║ spam      ║   41.9999 ║
╠═══════════╬═══════════╣
║ eggs      ║  451      ║
╚═══════════╩═══════════╝
""",
    "fancy_grid": """
╒═══════════╤═══════════╕
│ strings   │   numbers │
╞═══════════╪═══════════╡
│ spam      │   41.9999 │
├───────────┼───────────┤
│ eggs      │  451      │
╘═══════════╧═══════════╛
""",
}


@pytest.mark.parametrize("fmt", sorted(BOX_GRID_EXPECTED))
def test_box_drawing_grid_formats(fmt):
    assert tabulate(DATA, HEADERS, fmt) == block(BOX_GRID_EXPECTED[fmt])


OUTLINE_EXPECTED = {
    "outline": """
+-----------+-----------+
| strings   |   numbers |
+===========+===========+
| spam      |   41.9999 |
| eggs      |  451      |
+-----------+-----------+
""",
    "simple_outline": """
┌───────────┬───────────┐
│ strings   │   numbers │
├───────────┼───────────┤
│ spam      │   41.9999 │
│ eggs      │  451      │
└───────────┴───────────┘
""",
    "rounded_outline": """
╭───────────┬───────────╮
│ strings   │   numbers │
├───────────┼───────────┤
│ spam      │   41.9999 │
│ eggs      │  451      │
╰───────────┴───────────╯
""",
    "heavy_outline": """
┏━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ strings   ┃   numbers ┃
┣━━━━━━━━━━━╋━━━━━━━━━━━┫
┃ spam      ┃   41.9999 ┃
┃ eggs      ┃  451      ┃
┗━━━━━━━━━━━┻━━━━━━━━━━━┛
""",
    "mixed_outline": """
┍━━━━━━━━━━━┯━━━━━━━━━━━┑
│ strings   │   numbers │
┝━━━━━━━━━━━┿━━━━━━━━━━━┥
│ spam      │   41.9999 │
│ eggs      │  451      │
┕━━━━━━━━━━━┷━━━━━━━━━━━┙
""",
    "double_outline": """
╔═══════════╦═══════════╗
║ strings   ║   numbers ║
╠═══════════╬═══════════╣
║ spam      ║   41.9999 ║
║ eggs      ║  451      ║
╚═══════════╩═══════════╝
""",
    "fancy_outline": """
╒═══════════╤═══════════╕
│ strings   │   numbers │
╞═══════════╪═══════════╡
│ spam      │   41.9999 │
│ eggs      │  451      │
╘═══════════╧═══════════╛
""",
}


@pytest.mark.parametrize("fmt", sorted(OUTLINE_EXPECTED))
def test_outline_formats(fmt):
    assert tabulate(DATA, HEADERS, fmt) == block(OUTLINE_EXPECTED[fmt])


def test_pipe_and_github_formats():
    with_headers = block(
        """
| strings   |   numbers |
|:----------|----------:|
| spam      |   41.9999 |
| eggs      |  451      |
"""
    )
    without_headers = block(
        """
|:-----|---------:|
| spam |  41.9999 |
| eggs | 451      |
"""
    )
    assert tabulate(DATA, HEADERS, "pipe") == with_headers
    assert tabulate(DATA, tablefmt="pipe") == without_headers
    # github is an alias of pipe
    assert tabulate(DATA, HEADERS, "github") == with_headers
    assert tabulate(DATA, tablefmt="github") == without_headers


def test_pipe_escapes_pipe_characters():
    out = tabulate([["a|b", "c"]], ["h1", "h2"], tablefmt="pipe")
    assert "a\\|b" in out
    # the header separator line plus header plus one data row
    assert len(out.splitlines()) == 3


def test_orgtbl_format():
    with_headers = block(
        """
| strings   |   numbers |
|-----------+-----------|
| spam      |   41.9999 |
| eggs      |  451      |
"""
    )
    without_headers = block(
        """
| spam |  41.9999 |
| eggs | 451      |
"""
    )
    assert tabulate(DATA, HEADERS, "orgtbl") == with_headers
    assert tabulate(DATA, tablefmt="orgtbl") == without_headers


def test_rst_format():
    with_headers = block(
        """
=========  =========
strings      numbers
=========  =========
spam         41.9999
eggs        451
=========  =========
"""
    )
    without_headers = block(
        """
====  ========
spam   41.9999
eggs  451
====  ========
"""
    )
    assert tabulate(DATA, HEADERS, "rst") == with_headers
    assert tabulate(DATA, tablefmt="rst") == without_headers


def test_psql_format():
    expected = block(
        """
+-----------+-----------+
| strings   |   numbers |
|-----------+-----------|
| spam      |   41.9999 |
| eggs      |  451      |
+-----------+-----------+
"""
    )
    assert tabulate(DATA, HEADERS, "psql") == expected


def test_presto_format():
    expected = block(
        """
 strings   |   numbers
-----------+-----------
 spam      |   41.9999
 eggs      |  451
"""
    )
    assert tabulate(DATA, HEADERS, "presto") == expected


def test_jira_format():
    expected = block(
        """
|| strings   ||   numbers ||
| spam      |   41.9999 |
| eggs      |  451      |
"""
    )
    assert tabulate(DATA, HEADERS, "jira") == expected


def test_mediawiki_format():
    rows = [["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]]
    expected = block(
        """
{| class="wikitable" style="text-align: left;"
|+ <!-- caption -->
|-
! strings   !! style="text-align: right;"|   numbers
|-
| spam      || style="text-align: right;"|   41.9999
|-
| eggs      || style="text-align: right;"|  451
|}
"""
    )
    assert tabulate(rows, headers="firstrow", tablefmt="mediawiki") == expected


def test_html_format_with_header():
    rows = [["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]]
    out = tabulate(rows, headers="firstrow", tablefmt="html")
    lines = out.split("\n")
    assert lines[0] == "<table>"
    assert lines[1] == "<thead>"
    assert lines[2].startswith("<tr><th")
    assert lines[3] == "</thead>"
    assert lines[4] == "<tbody>"
    assert lines[-2] == "</tbody>"
    assert lines[-1] == "</table>"
    assert (
        lines[5]
        == '<tr><td style="text-align: left;">spam     </td>'
        '<td style="text-align: right;">  41.9999</td></tr>'
    )
    assert (
        lines[6]
        == '<tr><td style="text-align: left;">eggs     </td>'
        '<td style="text-align: right;"> 451     </td></tr>'
    )


def test_html_escapes_and_unsafehtml_does_not():
    rows = [["<b>x</b>", 1]]
    safe = tabulate(rows, tablefmt="html")
    unsafe = tabulate(rows, tablefmt="unsafehtml")
    assert safe == (
        "<table>\n"
        "<tbody>\n"
        '<tr><td style="text-align: left;">&lt;b&gt;x&lt;/b&gt;</td>'
        '<td style="text-align: right;">1</td></tr>\n'
        "</tbody>\n"
        "</table>"
    )
    assert unsafe == (
        "<table>\n"
        "<tbody>\n"
        '<tr><td style="text-align: left;"><b>x</b></td>'
        '<td style="text-align: right;">1</td></tr>\n'
        "</tbody>\n"
        "</table>"
    )


def test_html_output_is_a_jupyter_str_subclass():
    out = tabulate([[1, 2]], tablefmt="html")
    assert isinstance(out, str)
    assert out._repr_html_() == out
    assert out.str == out


LATEX_EXPECTED = {
    "latex": r"""
\begin{tabular}{lr}
\hline
 spam &  41.9999 \\
 eggs & 451      \\
\hline
\end{tabular}
""",
    "latex_booktabs": r"""
\begin{tabular}{lr}
\toprule
 spam &  41.9999 \\
 eggs & 451      \\
\bottomrule
\end{tabular}
""",
    "latex_longtable": r"""
\begin{longtable}{lr}
\hline
 spam &  41.9999 \\
 eggs & 451      \\
\hline
\end{longtable}
""",
    "latex_raw": r"""
\begin{tabular}{lr}
\hline
 spam &  41.9999 \\
 eggs & 451      \\
\hline
\end{tabular}
""",
}


@pytest.mark.parametrize("fmt", sorted(LATEX_EXPECTED))
def test_latex_family_formats(fmt):
    assert tabulate(DATA, tablefmt=fmt) == block(LATEX_EXPECTED[fmt])


def test_latex_escapes_special_characters_but_latex_raw_does_not():
    rows = [["a_b", "50%"]]
    escaped = tabulate(rows, tablefmt="latex")
    raw = tabulate(rows, tablefmt="latex_raw")
    assert escaped == "\\begin{tabular}{ll}\n\\hline\n a\\_b & 50\\% \\\\\n\\hline\n\\end{tabular}"
    assert raw == "\\begin{tabular}{ll}\n\\hline\n a_b & 50% \\\\\n\\hline\n\\end{tabular}"


def test_tsv_format():
    assert tabulate([["foo", 1], ["spam", 23]], tablefmt="tsv") == "foo \t 1\nspam\t23"


def test_colon_grid_format():
    default_align = block(
        """
+-----------+-----------+
| strings   | numbers   |
+:==========+:==========+
| spam      | 41.9999   |
+-----------+-----------+
| eggs      | 451       |
+-----------+-----------+
"""
    )
    right_left = block(
        """
+-----------+-----------+
| strings   | numbers   |
+==========:+:==========+
| spam      | 41.9999   |
+-----------+-----------+
| eggs      | 451       |
+-----------+-----------+
"""
    )
    assert tabulate(DATA, HEADERS, "colon_grid") == default_align
    assert tabulate(DATA, HEADERS, "colon_grid", colalign=["right", "left"]) == right_left


def test_pretty_format_centers_everything_as_strings():
    expected = block(
        """
+---------+---------+
| strings | numbers |
+---------+---------+
|  spam   | 41.9999 |
|  eggs   |  451.0  |
+---------+---------+
"""
    )
    assert tabulate(DATA, HEADERS, "pretty") == expected


def test_textile_format():
    assert tabulate([["a", 1], ["b", 22]], tablefmt="textile") == "|<. a  |>.  1 |\n|<. b  |>. 22 |"


def test_asciidoc_format():
    no_header = tabulate([["a", 1], ["b", 22]], tablefmt="asciidoc")
    assert no_header == '[cols="<3,>4"]\n|====\n| a |  1\n| b | 22\n|===='
    with_header = tabulate([["a", 1]], headers=["x", "y"], tablefmt="asciidoc")
    assert with_header == (
        '[cols="<5,>5",options="header"]\n'
        "|====\n"
        "| x   |   y\n"
        "| a   |   1\n"
        "|===="
    )


def test_moinmoin_format():
    out = tabulate([["a", 1]], tablefmt="moinmoin")
    assert out == '||<style="text-align: left;">  a  ||<style="text-align: right;">  1  ||'


def test_tabulate_formats_is_sorted_and_contains_the_documented_names():
    assert tabulate_formats == sorted(tabulate_formats)
    for name in [
        "plain",
        "simple",
        "grid",
        "fancy_grid",
        "github",
        "pipe",
        "orgtbl",
        "rst",
        "mediawiki",
        "html",
        "latex",
        "latex_raw",
        "latex_booktabs",
        "latex_longtable",
        "tsv",
        "psql",
        "jira",
        "presto",
        "pretty",
        "textile",
        "moinmoin",
        "unsafehtml",
        "asciidoc",
    ]:
        assert name in tabulate_formats
    assert len(set(tabulate_formats)) == len(tabulate_formats)


@pytest.mark.parametrize("fmt", tabulate_formats)
def test_every_format_renders_headers_and_data(fmt):
    out = tabulate([["a", 1], ["b", 2]], headers=["name", "n"], tablefmt=fmt)
    assert isinstance(out, str)
    assert out
    assert "name" in out
    # an empty table never raises either
    tabulate([], tablefmt=fmt)


def test_unknown_tablefmt_falls_back_to_simple():
    assert tabulate(DATA, HEADERS, "no_such_format") == tabulate(DATA, HEADERS, "simple")


def test_simple_separated_format():
    tsv = simple_separated_format("\t")
    assert tabulate([["foo", 1], ["spam", 23]], tablefmt=tsv) == "foo \t 1\nspam\t23"
    pipe = simple_separated_format("|")
    assert tabulate([["foo", 1], ["spam", 23]], tablefmt=pipe) == "foo | 1\nspam|23"
    semi = simple_separated_format(";")
    assert tabulate([["x", 1]], headers=["a", "b"], tablefmt=semi) == "a  ;  b\nx  ;  1"


def test_custom_table_format_object():
    fmt = TableFormat(
        lineabove=Line("<", "~", "+", ">"),
        linebelowheader=None,
        linebetweenrows=None,
        linebelow=Line("<", "~", "+", ">"),
        headerrow=DataRow("[", ";", "]"),
        datarow=DataRow("(", ",", ")"),
        padding=0,
        with_header_hide=None,
    )
    assert tabulate([[1, 2]], tablefmt=fmt) == "<~+~>\n(1,2)\n<~+~>"
