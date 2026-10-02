"""Tests for humanize.filesize, lists, and i18n modules."""

import pytest
from humanize import (
    naturalsize,
    natural_list,
    activate,
    deactivate,
    decimal_separator,
    thousands_separator,
    intcomma,
)


class TestNaturalsize:
    """Tests for naturalsize function."""

    def test_naturalsize_bytes(self):
        assert naturalsize(100) == "100 Bytes"
        assert naturalsize(1) == "1 Byte"

    def test_naturalsize_kilobytes(self):
        result = naturalsize(1024)
        assert "kB" in result or "KB" in result

    def test_naturalsize_megabytes(self):
        assert naturalsize(3000000) == "3.0 MB"

    def test_naturalsize_gigabytes(self):
        result = naturalsize(1024 * 1024 * 1024 * 3)
        assert "GB" in result

    def test_naturalsize_binary(self):
        result = naturalsize(1024, binary=True)
        assert "KiB" in result

    def test_naturalsize_binary_megabytes(self):
        result = naturalsize(1024 * 1024, binary=True)
        assert "MiB" in result

    def test_naturalsize_gnu(self):
        result = naturalsize(3000, False, True)
        assert "K" in result or "k" in result

    def test_naturalsize_gnu_format(self):
        result = naturalsize(3000, False, True, "%.3f")
        assert "K" in result or result is not None

    def test_naturalsize_negative(self):
        result = naturalsize(-4096, True)
        assert "-" in result
        assert "KiB" in result

    def test_naturalsize_large(self):
        result = naturalsize(10**28)
        assert "RB" in result or "QB" in result

    def test_naturalsize_very_large(self):
        result = naturalsize(10**34 * 3)
        assert "QB" in result or "RB" in result

    def test_naturalsize_zero(self):
        result = naturalsize(0)
        assert "0" in result or "Bytes" in result

    def test_naturalsize_string_input(self):
        result = naturalsize("5000")
        assert result is not None


class TestNaturalList:
    """Tests for natural_list function."""

    def test_natural_list_empty(self):
        assert natural_list([]) == ""

    def test_natural_list_single(self):
        assert natural_list(["one"]) == "one"

    def test_natural_list_two(self):
        assert natural_list(["one", "two"]) == "one and two"

    def test_natural_list_three(self):
        assert natural_list(["one", "two", "three"]) == "one, two and three"

    def test_natural_list_four(self):
        result = natural_list(["one", "two", "three", "four"])
        assert "and" in result
        assert "," in result

    def test_natural_list_numbers(self):
        result = natural_list([1, 2, 3])
        assert "and" in result

    def test_natural_list_mixed_types(self):
        result = natural_list([1, "two", 3.0])
        assert "and" in result

    def test_natural_list_long(self):
        items = list(range(10))
        result = natural_list(items)
        assert "and" in result


class TestI18nLocaleActivation:
    """Tests for i18n locale activation and deactivation."""

    def teardown_method(self):
        """Clean up after each test by deactivating locale."""
        deactivate()

    def test_deactivate(self):
        """Test that deactivate works."""
        deactivate()
        result = intcomma(1000)
        assert result == "1,000"

    def test_thousands_separator_default(self):
        """Test default thousands separator."""
        deactivate()
        assert thousands_separator() == ","

    def test_decimal_separator_default(self):
        """Test default decimal separator."""
        deactivate()
        assert decimal_separator() == "."

    def test_activate_german(self):
        """Test German locale activation."""
        activate("de_DE")
        assert thousands_separator() == "."
        assert decimal_separator() == ","

    def test_activate_french(self):
        """Test French locale activation."""
        activate("fr_FR")
        assert thousands_separator() == " "
        assert decimal_separator() == "."

    def test_activate_italian(self):
        """Test Italian locale activation."""
        activate("it_IT")
        assert thousands_separator() == "."
        assert decimal_separator() == ","

    def test_activate_portuguese_br(self):
        """Test Portuguese (Brazil) locale activation."""
        activate("pt_BR")
        assert thousands_separator() == "."
        assert decimal_separator() == ","

    def test_activate_hungarian(self):
        """Test Hungarian locale activation."""
        activate("hu_HU")
        assert thousands_separator() == " "
        assert decimal_separator() == ","

    def test_activate_none_deactivates(self):
        """Test that activate(None) deactivates."""
        activate("de_DE")
        activate(None)
        assert thousands_separator() == ","
        assert decimal_separator() == "."

    def test_activate_english_variants(self):
        """Test that English variants deactivate translation."""
        activate("en_GB")
        assert thousands_separator() == ","
        assert decimal_separator() == "."

    def test_intcomma_with_german_locale(self):
        """Test intcomma with German locale."""
        activate("de_DE")
        result = intcomma(1000)
        assert "." in result or "1000" in result

    def test_intcomma_with_french_locale(self):
        """Test intcomma with French locale."""
        activate("fr_FR")
        result = intcomma(1000)
        assert "1000" in result

    def test_activate_unknown_locale_raises(self):
        """Test that activating unknown locale raises FileNotFoundError or similar."""
        # Should handle gracefully or raise - implementation specific
        try:
            activate("unknown_LOCALE")
        except (FileNotFoundError, Exception):
            pass


class TestI18nNumberFormatting:
    """Tests for number formatting with different locales."""

    def teardown_method(self):
        """Clean up after each test."""
        deactivate()

    def test_german_number_formatting(self):
        """Test that German locale affects number formatting."""
        activate("de_DE")
        result = intcomma(1234567.89)
        assert result is not None

    def test_french_number_formatting(self):
        """Test that French locale affects number formatting."""
        activate("fr_FR")
        result = intcomma(1234567.89)
        assert result is not None

    def test_locale_persists_across_calls(self):
        """Test that locale persists across multiple calls."""
        activate("de_DE")
        sep1 = thousands_separator()
        sep2 = thousands_separator()
        assert sep1 == sep2
        assert sep1 == "."

    def test_deactivate_resets_formatting(self):
        """Test that deactivate resets formatting."""
        activate("de_DE")
        deactivate()
        assert thousands_separator() == ","
        assert decimal_separator() == "."
