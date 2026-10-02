"""Unit tests covering numeric formatting options of `tabulate()`:
`floatfmt`, `intfmt`, `missingval`, `disable_numparse`, `maxcolwidths`,
and `maxheadercolwidths`, plus general number-parsing behavior.
"""

from tabulate import tabulate


def test_floatfmt_single_spec_applies_to_all_columns():
    expected = "----  ----\n1.23  2.30\n----  ----"
    assert tabulate([[1.23456, 2.3]], floatfmt=".2f") == expected


def test_floatfmt_per_column_list():
    expected = "---  -----\n1.2  2.300\n---  -----"
    assert tabulate([[1.23456, 2.3]], floatfmt=(".1f", ".3f")) == expected


def test_intfmt_thousands_separator():
    expected = "---------\n1,000,000\n---------"
    assert tabulate([[1000000]], intfmt=",") == expected


def test_intfmt_only_applies_to_actual_int_type_values():
    # a numeric-looking string is not run through Python's int format-spec
    # machinery, so a comma intfmt has no effect on strings like "5"
    result = tabulate([["5"]], intfmt=",")
    assert "5" in result
    assert "," not in result


def test_missingval_custom_string():
    expected = "---  ---\na    N/A\nN/A    2\n---  ---"
    assert tabulate([["a", None], [None, 2]], missingval="N/A") == expected


def test_missingval_per_column_list():
    expected = "-  -\na  Y\nX  2\n-  -"
    assert tabulate([["a", None], [None, 2]], missingval=["X", "Y"]) == expected


def test_disable_numparse_true_keeps_leading_zeros():
    expected = "---  ---\n001  002\n---  ---"
    assert tabulate([["001", "002"]], disable_numparse=True) == expected


def test_disable_numparse_specific_columns():
    expected = "---  ---\n001  002\n---  ---"
    assert tabulate([["001", "002"]], disable_numparse=[0]) == expected


def test_numparse_quirk_scientific_looking_string():
    # documented quirk: a string like "42992e1" parses as a float/number
    expected = "------  -\n429920  1\n------  -"
    assert tabulate([["42992e1", 1]]) == expected


def test_bool_values_render_as_true_false_strings():
    expected = "----  -----\nTrue  False\n----  -----"
    assert tabulate([[True, False]]) == expected


def test_maxcolwidths_wraps_long_text():
    expected = (
        "+------------+\n"
        "| This is a  |\n"
        "| rather     |\n"
        "| long descr |\n"
        "| iption     |\n"
        "| that wraps |\n"
        "+------------+"
    )
    result = tabulate(
        [["This is a rather long description that wraps"]],
        maxcolwidths=[10],
        tablefmt="grid",
    )
    assert result == expected


def test_maxcolwidths_none_for_a_column_disables_wrapping():
    data = [("1", "John Smith", "A description that is long enough to wrap around")]
    result = tabulate(
        data,
        headers=("Issue Id", "Author", "Description"),
        maxcolwidths=[None, None, 20],
        tablefmt="grid",
    )
    # the un-wrapped columns keep their full single-line values
    assert "John Smith" in result
    for line in result.splitlines():
        if "John Smith" in line:
            assert "1" in line


def test_maxheadercolwidths_wraps_header_text():
    expected = (
        "+----------+\n"
        "|   A very |\n"
        "|     long |\n"
        "|   header |\n"
        "|     text |\n"
        "+==========+\n"
        "|        1 |\n"
        "+----------+"
    )
    result = tabulate(
        [[1]], headers=["A very long header text"], maxheadercolwidths=[10], tablefmt="grid"
    )
    assert result == expected


