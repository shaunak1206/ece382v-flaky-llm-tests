import humanize


# ordinal

def test_ordinal_basic_examples():
    assert humanize.ordinal(1) == "1st"
    assert humanize.ordinal(2) == "2nd"
    assert humanize.ordinal(3) == "3rd"
    assert humanize.ordinal(4) == "4th"
    assert humanize.ordinal(1002) == "1002nd"
    assert humanize.ordinal(103) == "103rd"


def test_ordinal_teens_use_th():
    assert humanize.ordinal(11) == "11th"
    assert humanize.ordinal(12) == "12th"
    assert humanize.ordinal(13) == "13th"
    assert humanize.ordinal(111) == "111th"


def test_ordinal_twenties_and_hundreds():
    assert humanize.ordinal(21) == "21st"
    assert humanize.ordinal(22) == "22nd"
    assert humanize.ordinal(101) == "101st"
    assert humanize.ordinal(100) == "100th"


def test_ordinal_non_numeric_returns_str():
    assert humanize.ordinal("something else") == "something else"
    assert humanize.ordinal([1, 2, 3]) == "[1, 2, 3]"
    assert humanize.ordinal("3") == "3rd"


def test_ordinal_gender_param_english():
    assert humanize.ordinal(1, gender="female") == "1st"
    assert humanize.ordinal(2, gender="male") == "2nd"


# intcomma

def test_intcomma_small_and_string():
    assert humanize.intcomma(100) == "100"
    assert humanize.intcomma("1000") == "1,000"


def test_intcomma_million():
    assert humanize.intcomma(1_000_000) == "1,000,000"


def test_intcomma_float():
    assert humanize.intcomma(1_234_567.25) == "1,234,567.25"


def test_intcomma_ndigits():
    assert humanize.intcomma(1234.5454545, 2) == "1,234.55"
    assert humanize.intcomma(14308.40, 1) == "14,308.4"
    assert humanize.intcomma("14308.40", 1) == "14,308.4"


def test_intcomma_none():
    assert humanize.intcomma(None) == "None"


def test_intcomma_negative():
    assert humanize.intcomma(-1234567) == "-1,234,567"


# intword

def test_intword_small_values_unchanged():
    assert humanize.intword("100") == "100"


def test_intword_thousand():
    assert humanize.intword("12400") == "12.4 thousand"


def test_intword_million():
    assert humanize.intword("1000000") == "1.0 million"


def test_intword_billion():
    assert humanize.intword(1_200_000_000) == "1.2 billion"


def test_intword_decillion():
    assert humanize.intword(8100000000000000000000000000000000) == "8.1 decillion"


def test_intword_invalid_input():
    assert humanize.intword(None) == "None"
    assert humanize.intword("foo") == "foo"


def test_intword_custom_format():
    assert humanize.intword("1234000", "%0.3f") == "1.234 million"


# apnumber

def test_apnumber_spelled_out_digits():
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for i, word in enumerate(words):
        assert humanize.apnumber(i) == word


def test_apnumber_ten_and_above_stay_digits():
    assert humanize.apnumber(10) == "10"
    assert humanize.apnumber(100) == "100"


def test_apnumber_string_input():
    assert humanize.apnumber("7") == "seven"


def test_apnumber_invalid_input():
    assert humanize.apnumber("foo") == "foo"
    assert humanize.apnumber(None) == "None"


# fractional

def test_fractional_fractions_and_mixed():
    assert humanize.fractional(0.3) == "3/10"
    assert humanize.fractional(1.3) == "1 3/10"
    assert humanize.fractional(float(1 / 3)) == "1/3"
    assert humanize.fractional(0.5) == "1/2"


def test_fractional_whole_number():
    assert humanize.fractional(1) == "1"


def test_fractional_invalid_input():
    assert humanize.fractional("ten") == "ten"
    assert humanize.fractional(None) == "None"


# scientific

def test_scientific_examples():
    assert humanize.scientific(float(0.3)) == "3.00 x 10⁻¹"
    assert humanize.scientific(int(500)) == "5.00 x 10²"


def test_scientific_negative():
    assert humanize.scientific(-1000) == "-1.00 x 10³"


def test_scientific_precision():
    assert humanize.scientific(1000, 1) == "1.0 x 10³"
    assert humanize.scientific(1000, 3) == "1.000 x 10³"


def test_scientific_string_input():
    assert humanize.scientific("99") == "9.90 x 10¹"


def test_scientific_invalid_input():
    assert humanize.scientific("foo") == "foo"
    assert humanize.scientific(None) == "None"


# clamp

def test_clamp_no_limits():
    assert humanize.clamp(123.456) == "123.456"


def test_clamp_below_floor():
    assert humanize.clamp(0.0001, floor=0.01) == "<0.01"


def test_clamp_ceil_boundary_and_above():
    assert humanize.clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
    assert humanize.clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"


def test_clamp_callable_format_and_token():
    assert humanize.clamp(1, format=humanize.intword, floor=1e6, floor_token="under ") == "under 1.0 million"


def test_clamp_none_returns_none():
    assert humanize.clamp(None) is None


# metric

def test_metric_kilo_with_unit():
    assert humanize.metric(1500, "V") == "1.50 kV"


def test_metric_mega():
    assert humanize.metric(2e8, "W") == "200 MW"


def test_metric_micro():
    assert humanize.metric(220e-6, "F") == "220 μF"


def test_metric_precision_and_fallback_to_scientific():
    assert humanize.metric(1e-14, precision=4) == "10.00 f"
    assert humanize.metric(1e40) == "1.00 x 10⁴⁰"
