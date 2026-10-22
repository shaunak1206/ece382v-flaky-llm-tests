import tabulate

T = tabulate.tabulate

TABLE = [["spam", 42], ["eggs", 451], ["bacon", 0]]
HEADERS = ["item", "qty"]


def test_doc_example_default_format():
    out = T([[1, 2.34], [-56, "8.999"], ["2", "10001"]])
    expected = "\n".join(
        [
            "---  ---------",
            "  1      2.34",
            "-56      8.999",
            "  2  10001",
            "---  ---------",
        ]
    )
    assert out == expected


def test_simple_with_headers():
    out = T(TABLE, HEADERS)
    expected = "\n".join(
        [
            "item      qty",
            "------  -----",
            "spam       42",
            "eggs      451",
            "bacon       0",
        ]
    )
    assert out == expected


def test_simple_without_headers_has_border_lines():
    out = T([["a"]])
    assert out == "-\na\n-"


def test_simple_with_headers_has_no_bottom_line():
    out = T([["a"]], headers=["h"])
    assert out == "h\n---\na"


def test_plain_without_headers():
    out = T([["a", 1], ["bb", 22]], tablefmt="plain")
    assert out == "a    1\nbb  22"


def test_grid_format():
    out = T(TABLE, HEADERS, tablefmt="grid")
    expected = "\n".join(
        [
            "+--------+-------+",
            "| item   |   qty |",
            "+========+=======+",
            "| spam   |    42 |",
            "+--------+-------+",
            "| eggs   |   451 |",
            "+--------+-------+",
            "| bacon  |     0 |",
            "+--------+-------+",
        ]
    )
    assert out == expected


def test_simple_grid_format():
    out = T(TABLE, HEADERS, tablefmt="simple_grid")
    expected = "\n".join(
        [
            "┌────────┬───────┐",
            "│ item   │   qty │",
            "├────────┼───────┤",
            "│ spam   │    42 │",
            "├────────┼───────┤",
            "│ eggs   │   451 │",
            "├────────┼───────┤",
            "│ bacon  │     0 │",
            "└────────┴───────┘",
        ]
    )
    assert out == expected


def test_rounded_grid_format():
    out = T(TABLE, HEADERS, tablefmt="rounded_grid")
    expected = "\n".join(
        [
            "╭────────┬───────╮",
            "│ item   │   qty │",
            "├────────┼───────┤",
            "│ spam   │    42 │",
            "├────────┼───────┤",
            "│ eggs   │   451 │",
            "├────────┼───────┤",
            "│ bacon  │     0 │",
            "╰────────┴───────╯",
        ]
    )
    assert out == expected


def test_heavy_grid_format():
    out = T(TABLE, HEADERS, tablefmt="heavy_grid")
    expected = "\n".join(
        [
            "┏━━━━━━━━┳━━━━━━━┓",
            "┃ item   ┃   qty ┃",
            "┣━━━━━━━━╋━━━━━━━┫",
            "┃ spam   ┃    42 ┃",
            "┣━━━━━━━━╋━━━━━━━┫",
            "┃ eggs   ┃   451 ┃",
            "┣━━━━━━━━╋━━━━━━━┫",
            "┃ bacon  ┃     0 ┃",
            "┗━━━━━━━━┻━━━━━━━┛",
        ]
    )
    assert out == expected


def test_mixed_grid_format():
    out = T(TABLE, HEADERS, tablefmt="mixed_grid")
    expected = "\n".join(
        [
            "┍━━━━━━━━┯━━━━━━━┑",
            "│ item   │   qty │",
            "┝━━━━━━━━┿━━━━━━━┥",
            "│ spam   │    42 │",
            "├────────┼───────┤",
            "│ eggs   │   451 │",
            "├────────┼───────┤",
            "│ bacon  │     0 │",
            "┕━━━━━━━━┷━━━━━━━┙",
        ]
    )
    assert out == expected


def test_double_grid_format():
    out = T(TABLE, HEADERS, tablefmt="double_grid")
    expected = "\n".join(
        [
            "╔════════╦═══════╗",
            "║ item   ║   qty ║",
            "╠════════╬═══════╣",
            "║ spam   ║    42 ║",
            "╠════════╬═══════╣",
            "║ eggs   ║   451 ║",
            "╠════════╬═══════╣",
            "║ bacon  ║     0 ║",
            "╚════════╩═══════╝",
        ]
    )
    assert out == expected


def test_fancy_grid_uses_box_drawing_edges():
    lines = T(TABLE, HEADERS, tablefmt="fancy_grid").splitlines()
    assert lines[0] == "╒════════╤═══════╕"
    assert lines[1] == "│ item   │   qty │"
    assert lines[2] == "╞════════╪═══════╡"
    assert lines[-1] == "╘════════╧═══════╛"


def test_pipe_format_marks_alignment_with_colons():
    lines = T(TABLE, HEADERS, tablefmt="pipe").splitlines()
    assert lines[0] == "| item   |   qty |"
    assert lines[1] == "|:-------|------:|"
    assert lines[2] == "| spam   |    42 |"
    assert len(lines) == 5


def test_rst_format():
    out = T(TABLE, HEADERS, tablefmt="rst")
    expected = "\n".join(
        [
            "======  =====",
            "item      qty",
            "======  =====",
            "spam       42",
            "eggs      451",
            "bacon       0",
            "======  =====",
        ]
    )
    assert out == expected


def test_org_format_header_separator():
    lines = T(TABLE, HEADERS, tablefmt="org").splitlines()
    assert lines[0] == "| item   |   qty |"
    assert lines[1] == "|--------+-------|"
    assert lines[2] == "| spam   |    42 |"


def test_github_format_rows_are_pipe_delimited():
    lines = T(TABLE, HEADERS, tablefmt="github").splitlines()
    assert "| item   |   qty |" in lines
    assert "| spam   |    42 |" in lines
    assert all(line.startswith("|") and line.endswith("|") for line in lines)


def test_html_format_basic_structure():
    out = T([["a", 1]], headers=["x", "y"], tablefmt="html")
    assert out.startswith("<table>")
    assert out.endswith("</table>")
    assert "<th>x</th>" in out
    assert "<td>a</td>" in out
    assert "<td" in out and ">1</td>" in out


def test_latex_format_column_spec_and_environment():
    out = T([["a", 1]], headers=["x", "y"], tablefmt="latex")
    assert "\\begin{tabular}{lr}" in out
    assert "\\end{tabular}" in out
    assert "\\hline" in out


def test_simple_separated_format_doc_example():
    tsv = tabulate.simple_separated_format("\t")
    out = T([["foo", 1], ["spam", 23]], tablefmt=tsv)
    assert out == "foo \t 1\nspam\t23"


def test_simple_separated_format_custom_separator():
    fmt = tabulate.simple_separated_format(",")
    out = T([["a", 1], ["bb", 22]], tablefmt=fmt)
    assert out == "a , 1\nbb,22"


def test_unknown_tablefmt_falls_back_to_simple():
    assert T(TABLE, HEADERS, tablefmt="no-such-format") == T(TABLE, HEADERS)


def test_tabulate_formats_lists_known_names():
    names = tabulate.tabulate_formats
    for name in ("simple", "plain", "grid", "pipe", "html", "latex"):
        assert name in names


def test_multiline_cell_in_grid():
    out = T([["a\nb", 1]], tablefmt="grid")
    expected = "\n".join(
        [
            "+---+---+",
            "| a | 1 |",
            "| b |   |",
            "+---+---+",
        ]
    )
    assert out == expected
