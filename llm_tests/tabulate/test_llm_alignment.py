"""Unit tests covering column/header alignment options of `tabulate()`:
`numalign`, `stralign`, `colalign`, `colglobalalign`, `headersalign`,
`headersglobalalign`, and `preserve_whitespace`.
"""

from tabulate import tabulate

DATA = [["a", 1], ["bb", 22]]


def test_numalign_right_is_default_like():
    # numbers default-align on the decimal point, which for pure ints
    # looks the same as right-alignment
    expected = "--  --\na    1\nbb  22\n--  --"
    assert tabulate(DATA, numalign="right") == expected


def test_numalign_center():
    expected = "--  --\na   1\nbb  22\n--  --"
    assert tabulate(DATA, numalign="center") == expected


def test_numalign_left():
    # with numalign="left", the single-digit "1" is padded with a
    # trailing space instead of a leading one (unlike numalign="right")
    expected = "--  --\na   1\nbb  22\n--  --"
    result = tabulate(DATA, numalign="left")
    assert result == expected
    assert result != tabulate(DATA, numalign="right")


def test_stralign_right():
    expected = "--  --\n a   1\nbb  22\n--  --"
    assert tabulate(DATA, stralign="right") == expected


def test_stralign_center():
    expected = "--  --\na    1\nbb  22\n--  --"
    assert tabulate(DATA, stralign="center") == expected


def test_colalign_overrides_per_column():
    expected = " s    n\n---  ---\n a    1\nbb   22"
    assert tabulate(DATA, headers=["s", "n"], colalign=("center", "center")) == expected


def test_colalign_partial_list_leaves_rest_default():
    # only override the first column; second column keeps its default numalign
    expected = "  s    n\n---  ---\n  a    1\n bb   22"
    result = tabulate(DATA, headers=["s", "n"], colalign=("right",))
    assert result == expected


def test_headersalign_override():
    expected = "  str  num\n-----  -----\na          1\nbb        22"
    result = tabulate(DATA, headers=["str", "num"], headersalign=("right", "left"))
    assert result == expected


def test_headersglobalalign():
    expected = " str    num\n-----  -----\na          1\nbb        22"
    result = tabulate(DATA, headers=["str", "num"], headersglobalalign="center")
    assert result == expected


def test_headersalign_same_follows_column_alignment():
    expected = " str    num\n-----  -----\n  a      1\n bb     22"
    result = tabulate(
        DATA, headers=["str", "num"], colalign=("center", "center"), headersalign=("same", "same")
    )
    assert result == expected


def test_preserve_whitespace_keeps_leading_spaces():
    data = [["  padded", 1]]
    result = tabulate(data, preserve_whitespace=True)
    assert "  padded" in result


def test_decimal_alignment_of_floats():
    data = [[1.5], [22.25], [3.125]]
    result = tabulate(data)
    lines = [line for line in result.splitlines() if line.strip("-")]
    # decimal points should line up in the same column
    decimal_positions = {line.index(".") for line in lines}
    assert len(decimal_positions) == 1


