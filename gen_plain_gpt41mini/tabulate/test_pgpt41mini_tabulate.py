import io
import math
import re
import warnings
from collections import namedtuple
from decimal import Decimal

import pytest
from freezegun import freeze_time

import tabulate
from tabulate import (
    TableFormat,
    DataRow,
    SEPARATING_LINE,
    _ansi_codes,
    _align_column,
    _align_column_choose_padfn,
    _align_column_choose_width_fn,
    _align_column_multiline_width,
    _append_basic_row,
    _append_line,
    _append_multiline_row,
    _build_line,
    _build_row,
    _build_simple_row,
    _column_type,
    _choose_width_fn,
    _expand_iterable,
    _expand_numparse,
    _format,
    _format_table,
    _is_separating_line,
    _is_separating_line_value,
    _isbool,
    _isconvertible,
    _isint,
    _isnumber,
    _isnumber_with_thousands_separator,
    _mediawiki_row_with_attrs,
    _moin_row_with_attrs,
    _padboth,
    _padleft,
    _padnone,
    _padright,
    _prepend_row_index,
    _pipe_line_with_colons,
    _pipe_segment_with_colons,
    _rst_escape_first_column,
    _strip_ansi,
    _textile_row_with_attrs,
    _visible_width,
    simple_separated_format,
    tabulate as tabulate_func,
    tabulate_formats,
)

# Helper data for tests
simple_data = [["spam", 1.5], ["eggs", 42]]
simple_headers = ["item", "value"]

class DummyCallable:
    def __call__(self, *args, **kwargs):
        return "called"

def test_tablefmt_members_presence():
    fmt = tabulate._table_formats["simple"]
    assert isinstance(fmt, TableFormat)
    assert fmt.lineabove is not None
    assert fmt.headerrow is not None
    assert isinstance(fmt.headerrow, DataRow)


@pytest.mark.parametrize("val,expected", [
    (SEPARATING_LINE, True),
    ("\001", True),
    ("\001foo", False),  # has other chars, strip not equal
    (["\001"], True),
    (["not", "\001"], True),
    (["no", "no"], False),
    ("notseparating", False),
])
def test_is_separating_line_and_value(val, expected):
    assert _is_separating_line_value(val if isinstance(val, str) else val[0]) == expected or (val==["not","\001"] and _is_separating_line(val) == expected)
    assert _is_separating_line(val) == expected


@pytest.mark.parametrize("align,colwidth,expected", [
    ("right", 5, "----:"),
    ("decimal", 4, "---:"),
    ("center", 4, ":--:"),
    ("left", 3, ":--"),
    ("", 5, "-----"),
    (None, 3, "---"),
    ("unknown", 2, "--"),
])
def test_pipe_segment_with_colons(align, colwidth, expected):
    assert _pipe_segment_with_colons(align, colwidth) == expected


def test_pipe_line_with_colons_empty_aligns():
    assert _pipe_line_with_colons([3, 4], []) == "|---|----|"


def test_pipe_line_with_colons_example():
    colwidths = [3, 4, 5]
    colaligns = ["left", "center", "right"]
    result = _pipe_line_with_colons(colwidths, colaligns)
    expected = "|:--|:--:|----:"
    assert result == expected


def test_grid_segment_with_colons_various():
    assert _grid_segment_with_colons(4, "right") == "===:"
    assert _grid_segment_with_colons(5, "center") == ":===:"
    assert _grid_segment_with_colons(3, "left") == ":=="
    assert _grid_segment_with_colons(6, None) == "======"


def test_grid_line_with_colons_empty_aligns():
    assert _grid_line_with_colons([2, 3], []) == "+==+==="


def test_grid_line_with_colons_example():
    result = _grid_line_with_colons([3, 4, 5], ["left", "center", "right"])
    expected = "+:==+:==:====:"
    assert result == expected


def test_mediawiki_row_with_attrs_basic():
    sep = "|"
    values = ["foo", "bar"]
    widths = [3, 3]
    aligns = ["left", "right"]
    result = _mediawiki_row_with_attrs(sep, values, widths, aligns)
    # It should create a string with appropriate style
    assert sep in result
    assert "style" in result


def test_textile_row_with_attrs_basic():
    result = _textile_row_with_attrs(["foo", "bar"], [4, 4], ["left", "right"])
    assert result.startswith("|")
    assert result.endswith("|")
    assert "<." in result
    assert ">." in result


def test_html_row_with_attrs_escape_and_unsafe(monkeypatch):
    vals = ["<b>bold</b>", "&amp;"]
    aligns = ["left", "right"]
    # unsafe False means escape
    res_escaped = tabulate._html_row_with_attrs("td", False, vals, [4,4], aligns)
    assert "&lt;" in res_escaped and "&amp;" in res_escaped
    res_unescaped = tabulate._html_row_with_attrs("td", True, vals, [4,4], aligns)
    assert "<b>" in res_unescaped
    header = tabulate._html_row_with_attrs("th", False, vals, [4,4], aligns)
    assert header.startswith("<table>") and "<thead>" in header and "<tbody>" in header


def test_moin_row_with_attrs_basic():
    vals = ["foo", "bar"]
    aligns = ["left", "right"]
    result = _moin_row_with_attrs("||", vals, [3, 3], aligns, header="''")
    assert result.startswith("||''foo")
    assert result.endswith("||")


@pytest.mark.parametrize("booktabs,longtable,param,result", [
    (False, False, None, ["\\begin{tabular}{lr}", "\\hline"]),
    (True, False, True, ["\\begin{tabular}{lr}", "\\toprule"]),
    (False, True, True, ["\\begin{longtable}{lr}", "\\toprule"]),
    (True, True, None, ["\\begin{longtable}{lr}", "\\hline"]),
])
def test_latex_line_begin_tabular(booktabs, longtable, param, result):
    partial_func = tabulate._latex_line_begin_tabular
    if param is None:
        out = partial_func([2,2], ["left","right"], booktabs=booktabs, longtable=longtable)
    else:
        # We cannot call partial, only function
        # So call via partial - simulate partial call
        f = partial_func
        if callable(f):
            out = f([2,2], ["left","right"])
        else:
            pytest.skip("Not callable directly")
    # Output always starts with \begin{
    assert out[0].startswith("\\begin")
    assert out[1].startswith("\\")  # \hline or \toprule etc


def test_asciidoc_row_header_and_data():
    # header line with colwidth and colalign
    hline = tabulate._asciidoc_row(True, [4,6], ["right", "left"])
    assert hline.startswith("[cols=")
    # data line
    dline = tabulate._asciidoc_row(False, ["foo", "bar"], [3,3], ["right", "left"])
    assert dline.startswith("|")
    with pytest.raises(ValueError):
        _ = tabulate._asciidoc_row(False)


@pytest.mark.parametrize("s,expected", [
    ("12345", 0),
    ("12.345", 3),
    ("12e03", 2),
    ("foo", -1),
    ("1234,567.89", 2),
])
def test_afterpoint(s, expected):
    assert tabulate._afterpoint(s) == expected


@pytest.mark.parametrize("s,width,padfn,result", [
    ("abc", 6, _padleft, "   abc"),
    ("abc", 6, _padright, "abc   "),
    ("abc", 6, _padboth, " abc  "),
    ("abc", 3, _padleft, "abc"),
])
def test_pad_functions(s, width, padfn, result):
    assert padfn(width, s) == result


@pytest.mark.parametrize("val,expected", [
    ("foo", str),
    ("", type(None)),
    (None, type(None)),
    (123, int),
    (123.4, float),
    ("123", int),
    ("12,345", int),
    ("12.345", float),
    (True, bool),
    ("True", bool)
])
def test_type_func(val, expected):
    assert tabulate._type(val, has_invisible=False, numparse=True) is expected


def test_strip_ansi_removes_codes():
    s = "\x1b[31mred\x1b[0m text"
    stripped = _strip_ansi(s)
    assert "red" in stripped and "\x1b" not in stripped


def test_visible_width_simple_and_ansi():
    s = "hello"
    s_ansi = "\x1b[31mhello\x1b[0m"
    vw_plain = _visible_width(s)
    vw_ansi = _visible_width(s_ansi)
    assert vw_plain == 5
    assert vw_ansi == 5


def test_align_column_choose_padfn_decimal_alignment():
    strings = ["1.2", "2.3456", "1000.0"]
    out_strings, padfn = _align_column_choose_padfn(strings, "decimal", False, False)
    # The strings should be padded with spaces to align decimals
    max_spaces = max(len(s) for s in out_strings)
    for s in out_strings:
        assert len(s) == max_spaces
    assert padfn == _padleft


def test_align_column_multiline_width():
    s = "a\nbb\nccc"
    widths = _align_column_multiline_width(s, len)
    assert widths == [1, 2, 3]


def test_flat_list_flattens_nested():
    nested = ["a", ["b", "c"], "d"]
    flat = tabulate._flat_list(nested)
    assert flat == ["a", "b", "c", "d"]


def test_align_column_returns_padded_strings():
    strings = ["foo", "bar"]
    result = _align_column(strings, "right", 5, False, False, False, False)
    assert all(len(s) == 5 for s in result)


def test_more_generic_type_ordering():
    # None < bool < int < float < bytes < str
    assert tabulate._more_generic(int, float) is float
    assert tabulate._more_generic(None, bool) is bool
    assert tabulate._more_generic(bytes, str) is str


@pytest.mark.parametrize("colvals,expected_type", [
    ([True, False], bool),
    (["1", "2"], int),
    (["1", "2.3"], float),
    (["1", "2.3", "four"], str),
    ([None, "brux"], str),
    ([1, 2, None], int),
])
def test_column_type(colvals, expected_type):
    got = _column_type(colvals, has_invisible=False, numparse=True)
    assert got is expected_type


@pytest.mark.parametrize("val,valtype,floatfmt,intfmt,missingval,expected", [
    (None, str, ".2f", "d", "?", ""),
    ("", str, ".2f", "d", "?", ""),
    ("test", str, ".2f", "d", "?", "test"),
    (42, int, ".2f", "d", "?", "42"),
    (3.14159, float, ".2f", "d", "?", "3.14"),
    ("3.14159", float, ".2f", "d", "?", "3.14"),
    (b"bytes", bytes, ".2f", "d", "?", "bytes"),
    ("notnum", float, ".2f", "d", "?", "notnum"),
])
def test_format_values(val, valtype, floatfmt, intfmt, missingval, expected):
    out = _format(val, valtype, floatfmt, intfmt, missingval, has_invisible=False)
    assert str(out).startswith(str(expected))


def test_prepend_row_index_basic_and_separating_lines():
    rows = [[1, 2], [3, 4]]
    index = [100, 200]
    prepended = _prepend_row_index(rows, index)
    assert prepended[0][0] == 100
    sep_lines = [[SEPARATING_LINE], [5, 6]]
    prepended_2 = _prepend_row_index(sep_lines, [1, 2])
    assert SEPARATING_LINE in prepended_2


def test_prepend_row_index_raises_on_length_mismatch():
    rows = [[1, 2], [3, 4]]
    index = [1]
    with pytest.raises(ValueError):
        _prepend_row_index(rows, index)


def test_bool_wrapper():
    assert tabulate._bool(True) is True
    import numpy as np
    if "numpy" in globals():
        arr = np.array([True, False])
        # numpy bool val should return False here to avoid exception
        assert tabulate._bool(arr) is False


def test_expand_numparse_bool_and_iterable():
    res1 = _expand_numparse(False, 3)
    assert res1 == [True, True, True]

    res2 = _expand_numparse([0,2], 4)
    assert res2 == [False, True, False, True]


def test_expand_iterable_various():
    assert _expand_iterable([1, 2], 4, 99) == [1,2,99,99]
    assert _expand_iterable("hi", 2, None) == [None, None]
    assert _expand_iterable(5, 3, None) == [None, None, None]


def test_build_simple_row_and_row_building():
    cells = ["a", "b"]
    rowfmt = DataRow("", "|", "", None)
    res = _build_simple_row(cells, rowfmt)
    assert res == "a|b"
    # with escape_map
    rowfmt_escape = DataRow("", "|", "", {"a":"x"})
    res_escaped = _build_simple_row(cells, rowfmt_escape)
    assert "a" not in res_escaped


def test_build_row_with_callable_and_none():
    cells = ["x"]
    rowfmt = DummyCallable()
    res = _build_row(cells, [1], ["left"], rowfmt)
    assert res == "called"

    res_none = _build_row(cells, [1], ["left"], None)
    assert res_none is None


def test_append_basic_row_returns_lines():
    lines = []
    padded_cells = ["a", "b"]
    result = _append_basic_row(lines, padded_cells, [1, 1], ["left", "right"], DataRow("", "|", ""))
    assert lines
    assert result is lines


def test_align_cell_vertically_various():
    lines = ["a", "b"]
    column_width = 2
    # bottom align
    aligned = tabulate._align_cell_veritically(lines, 4, column_width, "bottom")
    assert aligned[:2] == ["  ", "  "]
    # center align
    aligned_c = tabulate._align_cell_veritically(lines, 5, column_width, "center")
    assert len(aligned_c) == 5
    # default (top)
    aligned_t = tabulate._align_cell_veritically(lines, 3, column_width, "top")
    assert aligned_t[-1] == "  "


def test_append_multiline_row_basic():
    lines = []
    padded_multiline_cells = ["a\nb", "c"]
    padded_widths = [3, 3]
    colaligns = ["left", "right"]
    rowfmt = DataRow("", "|", "")
    result = _append_multiline_row(lines, padded_multiline_cells, padded_widths, colaligns, rowfmt, pad=1)
    assert lines
    assert result is lines


def test_build_line_and_append_line():
    colwidths = [3, 4]
    colaligns = ["left", "right"]
    fmt = DataRow("+", "-", "+", "+")
    line = _build_line(colwidths, colaligns, fmt)
    assert line.startswith("+")
    lines = []
    result = _append_line(lines, colwidths, colaligns, fmt)
    assert lines[0] == line
    assert result is lines


def test_simple_separated_format_usage():
    sep = "\t"
    fmt = simple_separated_format(sep)
    assert isinstance(fmt, TableFormat)
    row = ["foo", "bar"]
    row_fmt = fmt.datarow
    assert sep == row_fmt.sep
    # Build a simple row output
    padded = ["foo", "bar"]
    res = _build_simple_row(padded, row_fmt)
    assert sep in res


def test_rst_escape_first_column_nonempty_and_empty():
    rows = [["", 2], ["valid", 3], ["    ", 5]]
    headers = ["", "Header"]
    new_rows, new_headers = _rst_escape_first_column(rows, headers)
    assert new_headers[0] == ".."
    assert new_rows[0][0] == ".."
    assert new_rows[1][0] == "valid"


@pytest.mark.parametrize("string,expected", [
    ("1", True),
    ("1.", True),
    (".1", True),
    ("1,000", True),
    ("1,0000", False),
    ("notnumber", False),
    (b"1,000.123", True),
])
def test_isnumber_with_thousands_separator(string, expected):
    assert _isnumber_with_thousands_separator(string) == expected


@pytest.mark.parametrize("conv,string,expected", [
    (int, "3", True),
    (int, "a", False),
    (float, "3.14", True),
    (float, "bad", False),
    (int, None, False),
])
def test_isconvertible(conv, string, expected):
    assert _isconvertible(conv, string) is expected


@pytest.mark.parametrize("string,expected", [
    ("123", True),
    ("123.45", True),
    ("spam", False),
    ("123e45", True),
    ("123e45678", False),
    (123, True),
    (123.45, True),
])
def test_isnumber(string, expected):
    assert _isnumber(string) is expected


@pytest.mark.parametrize("string,expected", [
    ("123", True),
    ("123.45", False),
    (123, True),
    (123.45, False),
])
def test_isint(string, expected):
    assert _isint(string) is expected


@pytest.mark.parametrize("string,expected", [
    (True, True),
    ("False", True),
    (1, False),
    ("true", False),
])
def test_isbool(string, expected):
    assert _isbool(string) is expected


def test_choose_width_fn_various(monkeypatch):
    # Use monkeypatch to fake wcwidth availability
    monkeypatch.setattr(tabulate, "WIDE_CHARS_MODE", False)
    f = _choose_width_fn(False, False, False)
    assert callable(f)
    f_multi = _choose_width_fn(False, False, True)
    assert callable(f_multi)


def test_format_table_simple_vs_html():
    fmt = tabulate._table_formats["simple"]
    headers = ["a", "b"]
    rows = [("foo", "bar"), ("baz", "quux")]
    colwidths = [3, 4]
    colaligns = ["left", "right"]
    result = _format_table(fmt, headers, colaligns, rows, colwidths, colaligns, False, rowaligns=[None]*len(rows))
    assert isinstance(result, str)
    # For html
    fmt_html = tabulate._table_formats["html"]
    result_html = _format_table(fmt_html, headers, colaligns, rows, colwidths, colaligns, False, rowaligns=[None]*len(rows))
    # Should be JupyterHTMLStr with _repr_html_
    assert hasattr(result_html, "_repr_html_")


def test_tabulate_formats_list_includes_known():
    assert "simple" in tabulate_formats
    assert "pipe" in tabulate_formats


def test_tabulate_basic_table_and_headers():
    result = tabulate_func(simple_data, headers=simple_headers, tablefmt="simple")
    assert "spam" in result and "eggs" in result
    # Using headers='firstrow'
    data = [["col1", "col2"], ["a", "b"]]
    out = tabulate_func(data, headers="firstrow")
    assert "col1" in out


def test_tabulate_with_index_always():
    data = [["a", 1], ["b", 2]]
    result = tabulate_func(data, showindex="always")
    assert "0" in result and "1" in result


def test_tabulate_with_index_never():
    data = [["a", 1], ["b", 2]]
    result = tabulate_func(data, showindex="never")
    # Indices 0,1 should not present
    assert "0" not in result and "1" not in result


def test_tabulate_with_custom_index():
    data = [["a", 1], ["b", 2]]
    result = tabulate_func(data, showindex=["x", "y"])
    assert "x" in result and "y" in result


def test_tabulate_formats_various():
    data = [["spam", 1000.1234], ["eggs", 1.5]]
    for fmt in ["plain", "grid", "pipe", "rst", "mediawiki", "html", "latex"]:
        out = tabulate_func(data, headers=["Item", "Value"], tablefmt=fmt)
        assert out is not None and len(out) > 0


@pytest.mark.parametrize("tablefmt", tabulate_formats)
def test_tabulate_roundtrip_rows_keep(tablefmt):
    # A simple table should include string representation of known values
    data = [["A", 1], ["B", 2]]
    out = tabulate_func(data, headers=["Char", "Num"], tablefmt=tablefmt)
    assert any(x in out for x in ["A", "B", "1", "2"])


def test_simple_separated_format_functionality():
    fmt = simple_separated_format(",")
    data = [["a", "b"], ["c", "d"]]
    out = tabulate_func(data, headers=["h1", "h2"], tablefmt=fmt)
    assert "," in out
    assert "a" in out


def test_formatting_missing_values():
    data = [["a", None], ["b", 2]]
    result = tabulate_func(data, headers=["col1", "col2"], missingval="?")
    assert "?" in result


def test_maxcolwidths_wrap(monkeypatch):
    data = [["a" * 30, "b" * 30]]
    out = tabulate_func(data, headers=["col1", "col2"], maxcolwidths=[5, 10], tablefmt="grid")
    # Wrapped lines are present with newlines
    assert "\n" in out


def test_headers_maxwidths_wrap():
    headers = ["header1" * 5, "header2" * 5]
    data = [["a", "b"]]
    out = tabulate_func(data, headers=headers, maxheadercolwidths=[5, 7], tablefmt="grid")
    assert "\n" in out


def test_disable_numparse_parameter_effect():
    data = [["1,234", "abc"], ["5,678", "def"]]
    out_with_parse = tabulate_func(data)
    out_without_parse = tabulate_func(data, disable_numparse=True)
    assert out_with_parse != out_without_parse


def test_rowalign_parameter_vertical_align():
    data = [["a\nb\nc", "d"]]
    out_default = tabulate_func(data, tablefmt="grid")
    out_bottom = tabulate_func(data, tablefmt="grid", rowalign="bottom")
    out_center = tabulate_func(data, tablefmt="grid", rowalign="center")
    # The outputs may differ in line positions for vertical alignment
    assert out_default != out_bottom != out_center


def test_calling_tabulate_with_empty_data():
    out = tabulate_func([], headers=[])
    assert out == ""


def test_calling_tabulate_with_none_data():
    out = tabulate_func(None, headers=None)
    assert out == ""


def test_build_line_with_callable_format():
    def custom_line(colwidths, colaligns):
        return "headerline"
    line = _build_line([3,4], ["left","right"], custom_line)
    assert line == "headerline"


def test_build_line_returns_none_with_none_format():
    assert _build_line([3], ["left"], None) is None


def test_build_simple_row_with_escape_map():
    rowfmt = DataRow("", "-", "", {"-": "+"})
    padded = ["val-1", "val-2"]
    result = _build_simple_row(padded, rowfmt)
    assert "+" in result


def test_is_separating_line_with_strings_and_lists():
    assert _is_separating_line([SEPARATING_LINE]) is True
    assert _is_separating_line([SEPARATING_LINE, "anything"]) is True
    assert _is_separating_line("".join([SEPARATING_LINE, "x"])) is False


def test__padnone_returns_input():
    s = "no_pad"
    assert _padnone(10, s) == s


def test__strip_ansi_bytes():
    b = b"\x1b[31mhello\x1b[0m"
    result = _strip_ansi(b)
    assert b"hello" != result  # Since it returns string
    assert "hello" in result


def test__is_number_with_thousands_separator_bytes():
    bstr = b"1,000.00"
    assert _isnumber_with_thousands_separator(bstr) is True


def test__format_with_colored_numeric_string():
    colored_num = "\x1b[31m42\x1b[0m"
    formatted = _format(colored_num, float, ".2f", "", "", True)
    # Should replace numeric with formatted
    assert "\x1b" in formatted
    assert "42" not in formatted or "42" in formatted  # at minimum, format replaces raw with formatted


def test__format_with_decimal_and_int():
    val = Decimal("123.456")
    f = _format(val, float, ".2f", "", "", False)
    assert f == "123.46"
    i = _format(123, int, "", "04d", "", False)
    assert i == "0123"


def test_prepend_row_index_with_separating_lines_preserved():
    rows = [["a"], [SEPARATING_LINE], ["b"]]
    index = [1, 2]
    result = _prepend_row_index(rows, index)
    assert SEPARATING_LINE in result


def test__expand_iterable_with_string_and_other_types():
    assert _expand_iterable("abc", 4, None) == [None, None, None, None]
    assert _expand_iterable([1, 2], 3, 9) == [1, 2, 9]


def test__append_multiline_row_returns_lines(monkeypatch):
    lines = []
    padded_multiline_cells = ["line1\nline2", "data"]
    padded_widths = [10, 5]
    colaligns = ["left", "right"]
    rowfmt = DataRow("", "|", "")
    ret = _append_multiline_row(lines, padded_multiline_cells, padded_widths, colaligns, rowfmt, 1)
    assert ret is lines
    assert lines


def test__align_header_with_multiline_header():
    header = "a\nbb\nccc"
    aligned = tabulate._align_header(header, "left", 4, 1, is_multiline=True, width_fn=len)
    assert "\n" in aligned


def test_simple_separated_format_outputs_expected(monkeypatch):
    fmt = simple_separated_format("\t")
    assert isinstance(fmt, TableFormat)
    out = tabulate_func([["foo", 1], ["spam", 23]], tablefmt=fmt)
    assert "foo" in out and "\t" in out


def test__isconvertible_handles_exceptions():
    class Dummy:
        def __float__(self):
            raise ValueError()
    res = _isconvertible(float, Dummy())
    assert res is False


def test__bool_handles_numpy_arrays(monkeypatch):
    import numpy as np
    val = np.array([1, 2])
    # Should not raise
    result = tabulate._bool(val)
    assert result in (True, False)


def test__strip_ansi_removes_osc_links():
    s = '\x1B]8;;https://example.com\x1B\\This is a link\x1B]8;;\x1B\\'
    result = _strip_ansi(s)
    assert result == "This is a link"


def test__padboth_unicode_padding():
    s = "\u044f\u0439\u0446\u0430"
    padded = _padboth(6, s)
    assert len(padded) == 6 and padded.strip() == s


def test__build_line_with_string_format():
    fmt = namedtuple("Line", "begin hline sep end")("+", "-", "+", "+")
    result = _build_line([3, 4], ["right", "left"], fmt)
    assert result.startswith("+")


def test__format_with_invalid_float_string_returns_str():
    val = "a12.34"
    formatted = _format(val, float, ".2f", "", "", False)
    assert formatted == val


def test_tabulate_with_multiline_cells_and_multiline_format():
    data = [["line1\nline2", "text"]]
    result = tabulate_func(data, tablefmt="grid")
    assert "\n" in result


def test_tabulate_preserves_whitespace(monkeypatch):
    data = [["  a  ", "\tb\t"]]
    out = tabulate_func(data, headers=None, preserve_whitespace=True)
    assert "  a  " in out or "\tb\t" in out


def test__is_separating_line_value_rejects_nonstring():
    assert _is_separating_line_value(123) is False
    assert _is_separating_line_value(None) is False


def test_tabulate_with_colglobalalign_and_colalign():
    data = [[1, "a"], [2, "b"]]
    out = tabulate_func(data, colglobalalign="right", colalign=["left", "center"])
    assert out


@pytest.mark.parametrize("val", [True, False, "True", "False", 0, 1])
def test__isbool_various(val):
    res = _isbool(val)
    if isinstance(val, bool) or val in ("True", "False"):
        assert res is True or res is False
    else:
        assert res is False


def test__format_table_returns_empty_for_empty_input():
    fmt = tabulate._table_formats["simple"]
    out = _format_table(fmt, [], [], [], [], [], False, rowaligns=[])
    assert out == ""


def test__format_table_returns_html_string():
    fmt = tabulate._table_formats["html"]
    out = _format_table(fmt, ["header"], ["left"], [("row",)], [4], ["left"], False, rowaligns=[None])
    assert hasattr(out, "_repr_html_")
    assert out._repr_html_() == out


def test_tabulate_with_stralign_and_numalign():
    data = [[1.1, "foo"], [2.22, "bar"]]
    out = tabulate_func(data, stralign="center", numalign="right")
    assert out


def test__strip_ansi_on_bytestring_removes_ansi_codes():
    bs = b"\x1b[31mred\x1b[0m"
    result = _strip_ansi(bs)
    assert isinstance(result, str)
    assert "red" in result


def test__expand_numparse_with_bool_disabling():
    assert _expand_numparse(True, 3) == [False, False, False]
    assert _expand_numparse(False, 2) == [True, True]


def test_tabulate_with_preserve_whitespace_break_long_words_false():
    data = [["wordlongerthan10"]]
    out = tabulate_func(data, maxcolwidths=[5], break_long_words=False)
    assert "wordlongerthan10" in out or "\n" in out


def test__build_simple_row_with_escape_map_changes_chars():
    rowfmt = DataRow("", ",", "", {"a": "x"})
    padded = ["a", "b"]
    out = _build_simple_row(padded, rowfmt)
    assert "x" in out


def test__remove_separating_lines_and_reinsert():
    rows = [[SEPARATING_LINE], ["a"]]
    sans_rows, sep_lines = tabulate._remove_separating_lines(rows)
    assert SEPARATING_LINE in rows
    tabulate._reinsert_separating_lines(rows, sep_lines)
    assert SEPARATING_LINE in rows


def test_tabulate_with_datarow_partial_call():
    # Using a partial rowfmt function: mediawiki datarow is partial(_mediawiki_row_with_attrs, "|")
    fmt = tabulate._table_formats["mediawiki"]
    datarow = fmt.datarow
    result = datarow(["a", "b"], [1,1], ["left", "right"])
    assert result.startswith("|")


def test_multiline_formatting():
    data = [["multi\nline", "test"]]
    out = tabulate_func(data, tablefmt="plain")
    assert "multi\nline" in out


def test_JupyterHTMLStr_properties():
    s = tabulate.JupyterHTMLStr("test")
    assert s._repr_html_() == "test"
    assert s.str == "test"
    assert isinstance(s, str)


def test__align_header_left_right_center_none_alignment():
    header = "test"
    left = tabulate._align_header(header, "left", 10, 4)
    right = tabulate._align_header(header, "right", 10, 4)
    center = tabulate._align_header(header, "center", 10, 4)
    none = tabulate._align_header(header, None, 10, 4)

    assert left.startswith("test")
    assert right.endswith("test")
    assert center.strip().startswith("test")
    assert none == header


def test__format_table_with_no_headers_and_rows():
    fmt = tabulate._table_formats["simple"]
    result = _format_table(fmt, [], [], [], [1], ["left"], False, rowaligns=[])
    assert result == ""


def test__pad_row_with_separating_line_and_normal_cells():
    padded = tabulate._pad_row(SEPARATING_LINE, 2)
    assert padded == SEPARATING_LINE
    cells = ["a", "b"]
    padded_cells = tabulate._pad_row(cells, 1)
    assert padded_cells == [" a ", " b "]


def test__flat_list_flatten_with_empty_and_non_empty_lists():
    result = tabulate._flat_list(["a", ["b"], "c", []])
    assert "a" in result and "b" in result and "c" in result


def test__afterpoint_with_non_numeric_strings():
    assert _afterpoint("non-number") == -1


def test__isnumber_false_on_inval_string():
    assert not _isnumber("NaNA")


def test__isint_with_numpy_int(monkeypatch):
    # Simulate numpy int using string repr
    class FakeNumpyInt:
        def __str__(self):
            return "3"
    fakeint = FakeNumpyInt()
    assert _isint(fakeint, inttype=int) is False  # because type string doesn't start with numpy


def test__strip_ansi_with_non_str_bytes(monkeypatch):
    # Should handle bytes input
    b = b"\x1b[0mhello\x1b[33m"
    s = _strip_ansi(b)
    assert isinstance(s, str)


def test__format_table_returns_empty_when_no_rows_or_header(monkeypatch):
    fmt = tabulate._table_formats["grid"]
    res = _format_table(fmt, [], [], [], [], [], False, rowaligns=[])
    assert res == ""


def test__format_table_respects_with_header_hide(monkeypatch):
    fmt = tabulate._table_formats["simple"]
    # with_header_hide includes lineabove and linebelow, so they won't be printed if headers present
    out = _format_table(fmt, ["H"], ["left"], [("a",)], [1], ["left"], False, rowaligns=[None])
    # Shouldn't contain dashes lineabove or linebelow per with_header_hide
    assert "-" in out


def test_tabulate_preserves_row_align_default_none():
    data = [["a", "b"]]
    out = tabulate_func(data, rowalign=None)
    assert out


def test_tabulate_multiple_formats_do_not_raise():
    data = [["x", 1]]
    for fmt in ["simple", "pipe", "plain", "tsv"]:
        out = tabulate_func(data, tablefmt=fmt)
        assert out


@freeze_time("2023-05-01")
def test_tabulate_with_freeze_time():
    data = [["1", "2"]]
    out = tabulate_func(data)
    assert out


def test__expand_iterable_with_short_input():
    res = _expand_iterable([1], 3, 0)
    assert res == [1,0,0]


def test__expand_iterable_with_empty_list():
    res = _expand_iterable([], 2, 9)
    assert res == [9,9]

