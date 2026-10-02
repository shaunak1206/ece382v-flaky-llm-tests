"""Unit tests for humanize.filesize.naturalsize.

Based only on the public function signature and docstring in
subjects/humanize/src/humanize/filesize.py.
"""

import pytest

from humanize import filesize


@pytest.mark.parametrize(
    "value, binary, gnu, fmt, expected",
    [
        (3000000, False, False, "%.1f", "3.0 MB"),
        (300, False, True, "%.1f", "300B"),
        (3000, False, True, "%.1f", "2.9K"),
        (3000, False, True, "%.3f", "2.930K"),
        (3000, True, False, "%.1f", "2.9 KiB"),
        (10**28, False, False, "%.1f", "10.0 RB"),
        (10**34 * 3, False, False, "%.1f", "30000.0 QB"),
        (-4096, True, False, "%.1f", "-4.0 KiB"),
    ],
)
def test_naturalsize_docstring_examples(value, binary, gnu, fmt, expected):
    assert filesize.naturalsize(value, binary, gnu, fmt) == expected


def test_naturalsize_zero_bytes():
    assert filesize.naturalsize(0) == "0 Bytes"


def test_naturalsize_singular_byte():
    assert filesize.naturalsize(1) == "1 Byte"


def test_naturalsize_negative_singular_byte():
    assert filesize.naturalsize(-1) == "-1 Byte"


def test_naturalsize_decimal_default_uses_base_1000():
    # 1023 already crosses the decimal 1000 threshold, unlike binary mode.
    assert filesize.naturalsize(1023) == "1.0 kB"


def test_naturalsize_binary_below_1024_stays_in_bytes():
    assert filesize.naturalsize(1023, binary=True) == "1023 Bytes"


def test_naturalsize_binary_at_1024_promotes_to_kib():
    assert filesize.naturalsize(1024, binary=True) == "1.0 KiB"


def test_naturalsize_accepts_string_input():
    assert filesize.naturalsize("2048") == "2.0 kB"


def test_naturalsize_gnu_style_has_no_space_before_suffix():
    result = filesize.naturalsize(3000, gnu=True)
    assert " " not in result
    assert result.endswith("K")


def test_naturalsize_non_gnu_has_space_before_suffix():
    result = filesize.naturalsize(3000000)
    assert " " in result


def test_naturalsize_gnu_overrides_binary_flag():
    # When gnu=True, binary is ignored but the same base-1024 math is used.
    gnu_result = filesize.naturalsize(3000, binary=True, gnu=True)
    binary_result = filesize.naturalsize(3000, binary=True, gnu=False)
    assert gnu_result == "2.9K"
    assert binary_result == "2.9 KiB"
