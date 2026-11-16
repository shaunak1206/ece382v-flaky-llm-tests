import json
from dataclasses import dataclass, is_dataclass, fields
from datetime import datetime, date
from unittest.mock import patch

import pytest

from fistro.config import INT_LENGTH, INT_LIST_LENGTH, STR_LENGTH
from fistro.fistro import generate
from fistro import generators
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


@dataclass
class Simple:
    number: int
    text: str
    flag: bool


@dataclass
class WithDefault:
    number: int
    fixed: int = 5


FIXED_DATE = {
    'min_year': 2020,
    'max_year': 2020,
    'min_month': 5,
    'max_month': 5,
    'min_day': 10,
    'max_day': 10,
    'min_hour': 3,
    'max_hour': 3,
    'min_minute': 4,
    'max_minute': 4,
    'min_second': 5,
    'max_second': 5,
}


# ---------- generate ----------

def test_generate_returns_dataclass_type():
    result = generate(Simple)
    assert isinstance(result, type)
    assert is_dataclass(result)


def test_generate_keeps_class_name_and_field_names():
    result = generate(Simple)
    assert result.__name__ == 'Simple'
    assert [f.name for f in fields(result)] == ['number', 'text', 'flag']


def test_generate_instance_has_expected_types():
    instance = generate(Simple)()
    assert isinstance(instance.number, int)
    assert isinstance(instance.text, str)
    assert isinstance(instance.flag, bool)


def test_generate_as_dict():
    result = generate(Simple, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'number', 'text', 'flag'}
    assert isinstance(result['number'], int)
    assert isinstance(result['text'], str)
    assert isinstance(result['flag'], bool)


def test_generate_as_json_is_valid_json():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded.keys()) == {'number', 'text', 'flag'}


def test_generate_respects_default_values():
    instance = generate(WithDefault)()
    assert instance.fixed == 5
    assert isinstance(instance.number, int)


def test_generate_as_dict_respects_default_values():
    result = generate(WithDefault, as_dict=True)
    assert result['fixed'] == 5


def test_generate_with_custom_generator():
    def my_int() -> int:
        return 42

    instance = generate(WithDefault, generators=[my_int])()
    assert instance.number == 42
    assert instance.fixed == 5


def test_generate_str_field_length_default():
    @dataclass
    class OnlyStr:
        text: str

    instance = generate(OnlyStr)()
    assert isinstance(instance.text, str)


# ---------- int_generator ----------

def test_int_generator_default_range():
    for _ in range(50):
        value = int_generator()
        assert isinstance(value, int)
        assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_custom_length_range():
    for _ in range(50):
        value = int_generator(2)
        assert 0 <= value <= 900


def test_int_generator_length_zero():
    for _ in range(30):
        assert 0 <= int_generator(0) <= 9


def test_int_generator_calls_randint_with_bounds():
    with patch.object(generators, 'randint', return_value=7) as mocked:
        assert int_generator(3) == 7
    mocked.assert_called_once_with(0, 9 * 10 ** 3)


# ---------- other generators ----------

def test_bool_generator_returns_bool():
    values = {bool_generator() for _ in range(100)}
    assert values <= {True, False}
    assert all(isinstance(v, bool) for v in values)


def test_str_generator_length_and_population():
    value = str_generator(population='ab', length=20)
    assert len(value) == 20
    assert set(value) <= {'a', 'b'}


def test_str_generator_default_length():
    assert len(str_generator()) == STR_LENGTH


def test_datetime_generator_fixed_values():
    assert datetime_generator(FIXED_DATE) == datetime(2020, 5, 10, 3, 4, 5)


def test_date_generator_fixed_values():
    result = date_generator(FIXED_DATE)
    assert result == date(2020, 5, 10)
    assert type(result) is date


def test_int_list_generator_length_and_bounds():
    result = int_list_generator(list_length=6, int_length=1)
    assert len(result) == 6
    assert all(isinstance(i, int) and 0 <= i <= 90 for i in result)


def test_int_list_generator_default_length():
    assert len(int_list_generator()) == INT_LIST_LENGTH


def test_int_dict_generator_structure():
    result = int_dict_generator(dict_length=4, int_length=1)
    assert isinstance(result, dict)
    assert 1 <= len(result) <= 4
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


def test_default_generators_contents():
    result = default_generators()
    assert len(result) == 7
    assert int_generator in result
    assert all(callable(g) for g in result)
