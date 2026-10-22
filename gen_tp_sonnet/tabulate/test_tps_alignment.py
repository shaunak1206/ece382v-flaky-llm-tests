import tabulate

T = tabulate.tabulate


def test_numbers_right_aligned_by_default():
    assert T([[1], [100]], tablefmt="plain") == "  1\n100"


def test_numalign_left():
    assert T([[1], [100]], numalign="left", tablefmt="plain") == "1\n100"


def test_numalign_right_explicit():
    assert T([[1], [100]], numalign="right", tablefmt="plain") == "  1\n100"


def test_strings_left_aligned_by_default():
    assert T([["a"], ["ccc"]], tablefmt="plain") == "a\nccc"


def test_stralign_right():
    assert T([["a"], ["ccc"]], stralign="right", tablefmt="plain") == "  a\nccc"


def test_stralign_center():
    assert T([["a"], ["ccc"]], stralign="center", tablefmt="plain") == " a\nccc"


def test_colalign_per_column():
    out = T([["a", 1], ["bb", 22]], colalign=("right", "left"), tablefmt="plain")
    assert out == " a  1\nbb  22"


def test_colglobalalign_right():
    out = T([["a"], ["bbb"]], colglobalalign="right", tablefmt="plain")
    assert out == "  a\nbbb"


def test_decimal_alignment_of_floats():
    out = T([[1.5], [22.75]], tablefmt="plain")
    assert out == " 1.5\n22.75"


def test_mixed_int_and_float_column_aligns_integer_parts():
    out = T([[1], [22.5]], tablefmt="plain")
    assert out == " 1\n22.5"


def test_header_of_numeric_column_is_right_aligned():
    lines = T([[1], [100]], headers=["n"], tablefmt="plain").splitlines()
    assert lines[0] == "  n"


def test_header_of_string_column_is_left_aligned():
    lines = T([["abc"]], headers=["h"], tablefmt="plain").splitlines()
    assert lines[0] == "h"


def test_headersglobalalign_right():
    lines = T([["abc"]], headers=["h"], headersglobalalign="right", tablefmt="plain").splitlines()
    assert lines[0] == "  h"


def test_floatfmt_fixed_decimals():
    assert T([[1.23456]], floatfmt=".2f", tablefmt="plain") == "1.23"


def test_floatfmt_rounds_values():
    assert T([[1.26], [10.5]], floatfmt=".1f", tablefmt="plain") == " 1.3\n10.5"


def test_floatfmt_per_column_tuple():
    out = T([[1.0, 2.0]], floatfmt=(".1f", ".3f"), tablefmt="plain")
    assert out == "1.0  2.000"


def test_default_floatfmt_is_general_format():
    assert T([[0.1 + 0.2]], tablefmt="plain") == "0.3"


def test_intfmt_with_thousands_separator():
    assert T([[1234567]], intfmt=",", tablefmt="plain") == "1,234,567"


def test_numeric_strings_are_parsed_as_numbers():
    assert T([["1"], ["22"]], tablefmt="plain") == " 1\n22"


def test_numeric_float_string_is_reformatted():
    assert T([["1.50"]], tablefmt="plain") == "1.5"


def test_disable_numparse_keeps_original_string():
    assert T([["1.50"]], disable_numparse=True, tablefmt="plain") == "1.50"


def test_disable_numparse_for_selected_columns():
    out = T([["1.50", "2.50"]], disable_numparse=[0], tablefmt="plain")
    assert out.split() == ["1.50", "2.5"]


def test_scientific_notation_string_is_numeric():
    assert T([["1e3"]], tablefmt="plain") == "1000"


def test_large_integer_is_not_truncated():
    assert T([[10**20]], tablefmt="plain") == "100000000000000000000"


def test_nan_is_rendered():
    assert T([[float("nan")]], tablefmt="plain") == "nan"


def test_booleans_are_rendered_as_text():
    assert T([[True, False]], tablefmt="plain") == "True  False"
