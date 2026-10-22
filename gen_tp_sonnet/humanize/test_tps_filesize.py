import humanize


def test_naturalsize_decimal_default():
    assert humanize.naturalsize(3000000) == "3.0 MB"


def test_naturalsize_gnu_mode():
    assert humanize.naturalsize(300, False, True) == "300B"
    assert humanize.naturalsize(3000, False, True) == "2.9K"


def test_naturalsize_gnu_custom_format():
    assert humanize.naturalsize(3000, False, True, "%.3f") == "2.930K"


def test_naturalsize_binary():
    assert humanize.naturalsize(3000, True) == "2.9 KiB"


def test_naturalsize_very_large_values():
    assert humanize.naturalsize(10**28) == "10.0 RB"
    assert humanize.naturalsize(10**34 * 3) == "30000.0 QB"


def test_naturalsize_negative():
    assert humanize.naturalsize(-4096, True) == "-4.0 KiB"


def test_naturalsize_small_values():
    assert humanize.naturalsize(1) == "1 Byte"
    assert humanize.naturalsize(0) == "0 Bytes"
    assert humanize.naturalsize(300) == "300 Bytes"


def test_naturalsize_string_input():
    assert humanize.naturalsize("3000000") == "3.0 MB"


def test_naturalsize_unit_boundaries():
    assert humanize.naturalsize(1000) == "1.0 kB"
    assert humanize.naturalsize(1024, True) == "1.0 KiB"
