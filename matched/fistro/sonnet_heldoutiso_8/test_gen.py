import json
import string
from dataclasses import dataclass, is_dataclass, fields, asdict
from datetime import datetime, date

import pytest

from fistro import generators
from fistro.fistro import generate
from fistro.generators import (
    int_generator,
    bool_generator,
    str_generator,
    datetime_generator,
    date_generator,
    int_list_generator,
    int_dict_generator,
    default_generators,
)
from fistro.config import INT_LENGTH, STR_LENGTH, INT_LIST_LENGTH


@dataclass
class Simple:
    number: int
    text: str
    flag: bool


@dataclass
class WithDefault:
    number: int
    fixed: int = 7


# ---------------------------------------------------------------- generate


def test_generate_returns_dataclass_type():
    result = generate(Simple)
    assert isinstance(result, type)
    assert is_dataclass(result)


def test_generate_keeps_class_name():
    assert generate(Simple).__name__ == 'Simple'


def test_generate_keeps_field_names_and_types():
    result = generate(Simple)
    original = {f.name: f.type for f in fields(Simple)}
    generated = {f.name: f.type for f in fields(result)}
    assert generated == original


def test_generate_instance_has_values_of_right_types():
    instance = generate(Simple)()
    assert isinstance(instance.number, int)
    assert isinstance(instance.text, str)
    assert isinstance(instance.flag, bool)


def test_generate_as_dict_returns_dict_with_all_fields():
    result = generate(Simple, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'number', 'text', 'flag'}
    assert isinstance(result['number'], int)
    assert isinstance(result['text'], str)
    assert isinstance(result['flag'], bool)


def test_generate_as_json_returns_valid_json_string():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded.keys()) == {'number', 'text', 'flag'}


def test_generate_preserves_explicit_default():
    instance = generate(WithDefault)()
    assert instance.fixed == 7
    assert isinstance(instance.number, int)


def test_generate_as_dict_preserves_default():
    result = generate(WithDefault, as_dict=True)
    assert result['fixed'] == 7


def test_generate_with_custom_generator():
    def constant_int() -> int:
        return 42

    @dataclass
    class OnlyInt:
        value: int

    instance = generate(OnlyInt, generators=[constant_int])()
    assert instance.value == 42


def test_generate_fields_can_be_overridden_in_instance():
    instance = generate(Simple)(number=1, text='abc', flag=False)
    assert instance.number == 1
    assert instance.text == 'abc'
    assert instance.flag is False


def test_generate_as_dict_takes_precedence_over_json():
    result = generate(Simple, as_dict=True, as_json=True)
    assert isinstance(result, dict)


# ------------------------------------------------------------ int_generator


def test_int_generator_returns_int_in_default_range():
    for _ in range(50):
        value = int_generator()
        assert isinstance(value, int)
        assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_respects_length():
    for _ in range(50):
        assert 0 <= int_generator(2) <= 900


def test_int_generator_length_zero_gives_single_digit():
    for _ in range(50):
        assert 0 <= int_generator(0) <= 9


def test_int_generator_calls_randint_with_expected_bounds(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 5

    monkeypatch.setattr(generators, 'randint', fake_randint)
    assert int_generator(3) == 5
    assert calls == [(0, 9000)]


# ------------------------------------------------------- other generators


def test_bool_generator_returns_bool():
    values = {bool_generator() for _ in range(100)}
    assert values <= {True, False}
    assert all(isinstance(v, bool) for v in values)


def test_str_generator_length_and_population():
    result = str_generator(population='ab', length=20)
    assert len(result) == 20
    assert set(result) <= {'a', 'b'}
    assert len(str_generator()) == STR_LENGTH
    assert set(str_generator(length=30)) <= set(string.printable)


def test_datetime_generator_with_fixed_values():
    rand = {
        'min_year': 2000, 'max_year': 2000,
        'min_month': 3, 'max_month': 3,
        'min_day': 5, 'max_day': 5,
        'min_hour': 6, 'max_hour': 6,
        'min_minute': 7, 'max_minute': 7,
        'min_second': 8, 'max_second': 8,
    }
    assert datetime_generator(rand) == datetime(2000, 3, 5, 6, 7, 8)


def test_datetime_and_date_default_types():
    assert isinstance(datetime_generator(), datetime)
    result = date_generator()
    assert isinstance(result, date) and not isinstance(result, datetime)


def test_date_generator_with_fixed_values():
    rand = {
        'min_year': 2010, 'max_year': 2010,
        'min_month': 2, 'max_month': 2,
        'min_day': 4, 'max_day': 4,
    }
    assert date_generator(rand) == date(2010, 2, 4)


def test_int_list_generator_length_and_bounds():
    result = int_list_generator(list_length=5, int_length=1)
    assert len(result) == 5
    assert all(isinstance(i, int) and 0 <= i <= 90 for i in result)
    assert len(int_list_generator()) == INT_LIST_LENGTH


def test_int_dict_generator_structure():
    result = int_dict_generator(dict_length=4, int_length=1)
    assert isinstance(result, dict)
    assert 1 <= len(result) <= 4
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


def test_default_generators_contains_all_callables():
    result = default_generators()
    assert result == [
        int_generator,
        bool_generator,
        str_generator,
        datetime_generator,
        date_generator,
        int_list_generator,
        int_dict_generator,
    ]
    assert all(callable(g) for g in result)
