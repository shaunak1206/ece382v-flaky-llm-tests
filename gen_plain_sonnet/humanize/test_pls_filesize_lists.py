"""Unit tests for humanize.filesize and humanize.lists."""

import pytest

import humanize
from humanize import natural_list, naturalsize


@pytest.fixture(autouse=True)
def _reset_locale():
    humanize.deactivate()
    yield
    humanize.deactivate()


# naturalsize -----------------------------------------------------------


def test_naturalsize_small_values_are_bytes():
    assert naturalsize(0) == "0 Bytes"
    assert naturalsize(1) == "1 Byte"
    assert naturalsize(2) == "2 Bytes"
    assert naturalsize(999) == "999 Bytes"


def test_naturalsize_decimal_default():
    assert naturalsize(1000) == "1.0 kB"
    assert naturalsize(3_000_000) == "3.0 MB"
    assert naturalsize(1_500_000_000) == "1.5 GB"
    assert naturalsize(10**28) == "10.0 RB"


def test_naturalsize_binary_suffixes():
    assert naturalsize(1024, binary=True) == "1.0 KiB"
    assert naturalsize(3000, True) == "2.9 KiB"
    assert naturalsize(1024**2, binary=True) == "1.0 MiB"
    assert naturalsize(1023, binary=True) == "1023 Bytes"


def test_naturalsize_gnu_suffixes():
    assert naturalsize(300, False, True) == "300B"
    assert naturalsize(3000, False, True) == "2.9K"
    assert naturalsize(1, gnu=True) == "1B"
    assert naturalsize(1024**2, gnu=True) == "1.0M"


def test_naturalsize_custom_format():
    assert naturalsize(3000, False, True, "%.3f") == "2.930K"
    assert naturalsize(1500, format="%.2f") == "1.50 kB"
    assert naturalsize(1_000_000, format="%d") == "1 MB"


def test_naturalsize_negative_values():
    assert naturalsize(-4096, True) == "-4.0 KiB"
    assert naturalsize(-1) == "-1 Byte"
    assert naturalsize(-5000) == "-5.0 kB"


def test_naturalsize_accepts_strings_and_floats():
    assert naturalsize("3000000") == "3.0 MB"
    assert naturalsize(2500.0) == "2.5 kB"


def test_naturalsize_rounding_steps_up_to_next_suffix():
    assert naturalsize(999_999) == "1.0 MB"


def test_naturalsize_invalid_input_raises_value_error():
    with pytest.raises(ValueError):
        naturalsize("not a number")


# natural_list ----------------------------------------------------------


def test_natural_list_empty_and_single():
    assert natural_list([]) == ""
    assert natural_list(["one"]) == "one"
    assert natural_list([5]) == "5"


def test_natural_list_two_items():
    assert natural_list(["one", "two"]) == "one and two"


def test_natural_list_three_or_more_items():
    assert natural_list(["one", "two", "three"]) == "one, two and three"
    assert natural_list(["a", "b", "c", "d"]) == "a, b, c and d"


def test_natural_list_converts_items_to_str():
    assert natural_list([1, 2, 3]) == "1, 2 and 3"
    assert natural_list([None, 1.5]) == "None and 1.5"
    assert natural_list(("x", "y", "z")) == "x, y and z"
