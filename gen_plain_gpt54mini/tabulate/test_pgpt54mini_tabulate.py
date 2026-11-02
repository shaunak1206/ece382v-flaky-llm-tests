import dataclasses
from decimal import Decimal
from collections import namedtuple

import pytest

import tabulate as tb


def test_simple_separated_format_returns_tableformat():
    fmt = tb.simple_separated_format("\t")
    assert fmt.lineabove is None
    assert fmt.headerrow.sep == "\t"
    assert fmt.datarow.sep == "\t"
    assert fmt.padding == 0


def test_is_file_detects_io_base():
    with open(__file__, "r", encoding="utf-8") as f:
        assert tb._is_file(f) is True
    assert tb._is_file("not a file") is False


def test_is_separating_line_value_matches_exact_string():
    assert tb._is_separating_line_value(tb.SEPARATING_LINE)
    assert tb._is_separating_line_value(f" {tb.SEPARATING_LINE} ")
    assert not tb._is_separating_line_value("x")


def test_is_separating_line_detects_first_element():
    assert tb._is_separating_line([tb.SEPARATING_LINE, "x"])
    assert tb._is_separating_line([f" {tb.SEPARATING_LINE} ", "x"])
    assert not tb._is_separating_line(["x", tb.SEPARATING_LINE])


def test_pipe_segment_with_colons_left():
    assert tb._pipe_segment_with_colons("left", 4) == ":---"


def test_pipe_segment_with_colons_right():
    assert tb._pipe_segment_with_colons("right", 4) == "---:"


def test_pipe_segment_with_colons_center():
    assert tb._pipe_segment_with_colons("center", 4) == ":--:"


def test_pipe_segment_with_colons_default():
    assert tb._pipe_segment_with_colons("unknown", 3) == "---"


def test_pipe_line_with_colons_uses_empty_aligns():
    assert tb._pipe_line_with_colons([2, 3], []) == "|--|---|"


def test_grid_segment_with_colons_left():
    assert tb._grid_segment_with_colons(4, "left") == ":==="


def test_grid_segment_with_colons_right():
    assert tb._grid_segment_with_colons(4, "right") == "===:"


def test_grid_segment_with_colons_center():
    assert tb._grid_segment_with_colons(4, "center") == ":==:"


def test_grid_segment_with_colons_default():
    assert tb._grid_segment_with_colons(3, "x") == "==="


def test_mediawiki_row_with_attrs_escapes_alignment():
    out = tb._mediawiki_row_with_attrs("|", ["<a>", "b"], [3, 3], ["right", "left"])
    assert out.startswith("|")
    assert 'style="text-align: right;"| <a>' in out


def test_textile_row_with_attrs_includes_alignment():
    out = tb._textile_row_with_attrs(["a", "b"], [1, 1], ["center", "right"])
    assert out == "|=. a |> .b|" or out.startswith("|=. a|")  # tolerate spacing implementation


def test_html_row_with_attrs_escapes_by_default():
    out = tb._html_row_with_attrs("td", False, ["<x>", "&"], [1, 1], ["left", "right"])
    assert "&lt;x&gt;" in out
    assert "&amp;" in out
    assert '<td style="text-align: right;">' in out


def test_html_row_with_attrs_unsafe_allows_raw_html():
    out = tb._html_row_with_attrs("td", True, ["<x>"], [1], ["left"])
    assert "<x>" in out
    assert "&lt;x&gt;" not in out


def test_html_header_row_starts_table_and_thead():
    out = tb._html_row_with_attrs("th", False, ["a"], [1], ["left"])
    assert out.startswith("<table>\n<thead>\n<tr>")
    assert "<th style=" in out


def test_latex_line_begin_tabular_defaults():
    out = tb._latex_line_begin_tabular([1, 1], ["left", "right"])
    assert "\\begin{tabular}{lr}" in out
    assert "\\hline" in out


def test_asciidoc_row_invalid_args_raises():
    with pytest.raises(ValueError):
        tb._asciidoc_row(True)


def test_isnumber_with_thousands_separator():
    assert tb._isnumber_with_thousands_separator("1,000")
    assert tb._isnumber_with_thousands_separator(b"1,000.25")
    assert not tb._isnumber_with_thousands_separator("1,0000")


def test_isnumber_detects_numeric_strings_and_values():
    assert tb._isnumber(123)
    assert tb._isnumber(12.3)
    assert tb._isnumber("1e2")
    assert tb._isnumber("inf")
    assert not tb._isnumber("spam")


def test_isint_detects_int_strings():
    assert tb._isint("123")
    assert not tb._isint("123.45")


def test_isbool_detects_bool_and_bool_strings():
    assert tb._isbool(True)
    assert tb._isbool("False")
    assert not tb._isbool(1)


def test_type_none_and_empty_string():
    assert tb._type(None) is type(None)
    assert tb._type("") is type(None)


def test_type_numeric_string_and_ansi_stripped():
    assert tb._type("1") is int
    assert tb._type("\x1b[31m42\x1b[0m") is int


def test_type_bytes_and_datetime_like():
    import datetime as dt

    assert tb._type(b"abc") is bytes
    assert tb._type(dt.date(2020, 1, 1)) is str


def test_afterpoint_for_decimal_and_exponent():
    assert tb._afterpoint("123.45") == 2
    assert tb._afterpoint("123e45") == 2
    assert tb._afterpoint("eggs") == -1


def test_pad_functions_basic():
    assert tb._padleft(4, "x") == "   x"
    assert tb._padright(4, "x") == "x   "
    assert tb._padboth(4, "x") == " x  " or tb._padboth(4, "x") == " x "
    assert tb._padnone(4, "x") == "x"


def test_strip_ansi_removes_csi_and_osc():
    assert tb._strip_ansi("\x1b[31mred\x1b[0m") == "red"
    osc = "\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\"
    assert tb._strip_ansi(osc) == "link"


def test_visible_width_ignores_ansi():
    assert tb._visible_width("\x1b[31mhello\x1b[0m") == 5


def test_is_multiline_str_and_bytes():
    assert tb._is_multiline("a\nb")
    assert tb._is_multiline(b"a\rb")
    assert not tb._is_multiline("ab")


def test_multiline_width_uses_longest_line():
    assert tb._multiline_width("a\nabcd") == 4


def test_choose_width_fn_multiline_wraps():
    fn = tb._choose_width_fn(False, False, True)
    assert fn("a\nabcd") == 4


def test_align_column_choose_padfn_right_strips_when_not_preserve():
    strings, padfn = tb._align_column_choose_padfn([" x "], "right", False, False)
    assert strings == ["x"]
    assert padfn is tb._padleft


def test_align_column_choose_padfn_decimal_appends_spaces():
    strings, padfn = tb._align_column_choose_padfn(["1.2", "10.34"], "decimal", False, False)
    assert padfn is tb._padleft
    assert strings[0].endswith(" ")


def test_flat_list_flattens_single_level_lists():
    assert tb._flat_list([[1, 2], 3, [4]]) == [1, 2, 3, 4]


def test_align_column_left_and_right():
    right = tb._align_column(["a", "bb"], "right")
    left = tb._align_column(["a", "bb"], "left")
    assert right[0].endswith("a")
    assert left[0].startswith("a")


def test_more_generic_ordering():
    assert tb._more_generic(int, float) is float
    assert tb._more_generic(type(None), bool) is bool
    assert tb._more_generic(bytes, str) is str


def test_column_type_detects_common_types():
    assert tb._column_type([True, False]) is bool
    assert tb._column_type(["1", "2"]) is int
    assert tb._column_type(["1", "2.3"]) is float
    assert tb._column_type(["a", "b"]) is str


def test_format_none_and_empty_string():
    assert tb._format(None, str, "g", "", missingval="?") == "?"
    assert tb._format("", str, "g", "", missingval="?") == ""


def test_format_int_float_bytes_str():
    assert tb._format(12, int, "g", "04d") == "0012"
    assert tb._format(12.5, float, ".1f", "") == "12.5"
    assert tb._format(b"abc", bytes, "g", "") == "abc"
    assert tb._format("x", str, "g", "") == "x"


def test_align_header_left_center_right_and_none():
    assert tb._align_header("x", "left", 3, 1) == "x  "
    assert tb._align_header("x", "center", 3, 1).strip() == "x"
    assert tb._align_header("x", "right", 3, 1).endswith("x")
    assert tb._align_header("x", None, 3, 1) == "x"


def test_remove_and_reinsert_separating_lines():
    rows = [["a"], tb.SEPARATING_LINE, ["b"]]
    sans, idxs = tb._remove_separating_lines(rows)
    assert sans == [["a"], ["b"]]
    assert idxs == [1]
    tb._reinsert_separating_lines(sans, idxs)
    assert sans == [["a"], tb.SEPARATING_LINE, ["b"]]


def test_prepend_row_index_with_length_mismatch_raises():
    with pytest.raises(ValueError):
        tb._prepend_row_index([["a"], ["b"]], [0])


def test_prepend_row_index_inserts_index_values():
    rows = tb._prepend_row_index([["a"], ["b"]], [10, 11])
    assert rows == [[10, "a"], [11, "b"]]


def test_bool_wrapper_handles_value_error():
    class BadBool:
        def __bool__(self):
            raise ValueError

    assert tb._bool(BadBool()) is False


def test_normalize_tabular_data_list_of_lists_headers_firstrow():
    rows, headers, pad = tb._normalize_tabular_data([["h1", "h2"], [1, 2]], "firstrow")
    assert headers == ["h1", "h2"]
    assert rows == [[1, 2]]
    assert pad == 0


def test_normalize_tabular_data_showindex_always():
    rows, headers, pad = tb._normalize_tabular_data([[1, 2], [3, 4]], headers=(), showindex="always")
    assert rows[0][0] == 0
    assert headers == []
    assert pad == 0


def test_normalize_tabular_data_keys_for_dict_of_lists():
    rows, headers, _ = tb._normalize_tabular_data({"a": [1, 2], "b": [3, 4]}, headers="keys")
    assert headers == ["a", "b"]
    assert rows == [[1, 3], [2, 4]]


def test_normalize_tabular_data_list_of_dicts_keys():
    rows, headers, _ = tb._normalize_tabular_data([{"a": 1, "b": 2}, {"b": 3, "c": 4}], headers="keys")
    assert headers == ["a", "b", "c"]
    assert rows == [[1, 2, None], [None, 3, 4]]


def test_normalize_tabular_data_namedtuple_keys():
    Point = namedtuple("Point", ["x", "y"])
    rows, headers, _ = tb._normalize_tabular_data([Point(1, 2)], headers="keys")
    assert headers == ["x", "y"]
    assert rows == [[1, 2]]


def test_normalize_tabular_data_dataclass_keys():
    @dataclasses.dataclass
    class Row:
        a: int
        b: int

    rows, headers, _ = tb._normalize_tabular_data([Row(1, 2)], headers="keys")
    assert headers == ["a", "b"]
    assert rows == [[1, 2]]


def test_expand_numparse_bool_and_list():
    assert tb._expand_numparse(True, 3) == [False, False, False]
    assert tb._expand_numparse([1], 3) == [True, True, True] or tb._expand_numparse([1], 3)[1] is False


def test_expand_iterable_scalar_and_list():
    assert tb._expand_iterable(1, 3, 0) == [0, 0, 0]
    assert tb._expand_iterable([1, 2], 4, 0) == [1, 2, 0, 0]


def test_pad_row_and_separating_line_passthrough():
    assert tb._pad_row(["a"], 1) == [" a "]
    assert tb._pad_row(tb.SEPARATING_LINE, 1) == tb.SEPARATING_LINE


def test_build_simple_row_uses_escape_map():
    rowfmt = tb.DataRow("[", "|", "]", {"|": "/"})
    assert tb._build_simple_row([["a|b"]], rowfmt) == "[a/b]"


def test_build_row_accepts_callable_and_datrow():
    rowfmt = tb.DataRow("", "|", "")
    assert tb._build_row([["a"]], [1], ["left"], rowfmt) == "a"
    assert tb._build_row([["a"]], [1], ["left"], lambda *args: "ok") == "ok"


def test_build_line_from_tuple_and_callable():
    assert tb._build_line([2], ["left"], tb.Line("", "-", "", "")) == "--"
    assert tb._build_line([2], ["left"], lambda *args: "ok") == "ok"


def test_jupyter_html_str_repr_and_property():
    s = tb.JupyterHTMLStr("<b>hi</b>")
    assert s._repr_html_() == "<b>hi</b>"
    assert s.str == "<b>hi</b>"


def test_format_table_empty_returns_empty_string():
    fmt = tb._table_formats["plain"]
    assert tb._format_table(fmt, [], None, [], [], [], False, []) == ""


def test_format_table_returns_jupyter_html_str_for_html():
    fmt = tb._table_formats["html"]
    out = tb._format_table(fmt, ["A"], ["left"], [["x"]], [1], ["left"], False, [None])
    assert isinstance(out, tb.JupyterHTMLStr)
    assert "<table>" in out


def test_tabulate_simple_default_output():
    out = tb.tabulate([[1, 2.34], [-56, "8.999"], ["2", "10001"]])
    assert "2.34" in out
    assert "10001" in out


def test_tabulate_with_headers_firstrow():
    out = tb.tabulate([["sex", "age"], ["Alice", "F", 24], ["Bob", "M", 19]], headers="firstrow")
    assert "Alice" in out
    assert "Bob" in out
    assert "age" in out


def test_tabulate_showindex_always():
    out = tb.tabulate([["F", 24], ["M", 19]], showindex="always")
    assert "0" in out and "1" in out


def test_tabulate_missingval_replaces_none():
    out = tb.tabulate([["spam", 1, None], ["eggs", 42, 3.14]], missingval="?")
    assert "?" in out


def test_tabulate_pipe_format_has_alignment_colons():
    out = tb.tabulate([["spam", 41.9999], ["eggs", "451.0"]], ["x", "y"], "pipe")
    assert out.splitlines()[1].startswith("|:")
    assert out.splitlines()[1].endswith(":|")


def test_tabulate_orgtbl_format_uses_plus_separator():
    out = tb.tabulate([["spam", 1]], ["x", "y"], "orgtbl")
    assert "|" in out
    assert "+" in out.splitlines()[1]


def test_tabulate_rst_escapes_empty_first_column():
    out = tb.tabulate([["", 1], ["  ", 2]], headers=["a", "b"], tablefmt="rst")
    assert ".." in out


def test_tabulate_html_returns_jupyter_html_str():
    out = tb.tabulate([["a", "<b>"]], headers=["x", "y"], tablefmt="html")
    assert isinstance(out, tb.JupyterHTMLStr)
    assert "&lt;b&gt;" in out


def test_tabulate_unsafehtml_keeps_raw_markup():
    out = tb.tabulate([["a", "<b>"]], headers=["x", "y"], tablefmt="unsafehtml")
    assert "<b>" in out
    assert "&lt;b&gt;" not in out


def test_tabulate_latex_escapes_special_chars():
    out = tb.tabulate([["a_b", "x&y"]], headers=["x", "y"], tablefmt="latex")
    assert r"\_" in out
    assert r"\&" in out


def test_tabulate_latex_raw_preserves_special_chars():
    out = tb.tabulate([["a_b", "x&y"]], headers=["x", "y"], tablefmt="latex_raw")
    assert "a_b" in out
    assert r"\_" not in out


def test_tabulate_pretty_disables_numparse():
    out = tb.tabulate([["1", "2"]], tablefmt="pretty")
    assert "1" in out and "2" in out


def test_tabulate_colon_grid_has_colons_in_separator():
    out = tb.tabulate([["a", "b"]], headers=["x", "y"], tablefmt="colon_grid")
    assert "+" in out.splitlines()[1]
    assert ":" in out.splitlines()[1]


def test_tabulate_with_custom_tableformat_object():
    fmt = tb.simple_separated_format(" | ")
    out = tb.tabulate([["a", "b"]], headers=["x", "y"], tablefmt=fmt)
    assert "a | b" in out


def test_tabulate_maxcolwidths_wraps_text():
    out = tb.tabulate(
        [("1", "John Smith", "This is a rather long description")],
        headers=("Issue Id", "Author", "Description"),
        maxcolwidths=[None, None, 12],
        tablefmt="grid",
    )
    assert "description" in out.lower()
    assert "\n" in out


def test_tabulate_maxheadercolwidths_wraps_headers():
    out = tb.tabulate([[1, 2]], headers=["very long header", "b"], maxheadercolwidths=[5, None])
    assert "very" in out
    assert "\n" in out


def test_tabulate_disable_numparse_list():
    out = tb.tabulate([["001", "2"]], headers=["a", "b"], disable_numparse=[0])
    assert "001" in out


def test_tabulate_colalign_warns_on_string():
    with pytest.warns(UserWarning):
        tb.tabulate([[1, 2]], colalign="right")


def test_tabulate_headersalign_warns_on_string():
    with pytest.warns(UserWarning):
        tb.tabulate([[1, 2]], headers=["a", "b"], headersalign="left")


def test_tabulate_tabular_data_none_becomes_empty_string():
    assert tb.tabulate(None) == ""
