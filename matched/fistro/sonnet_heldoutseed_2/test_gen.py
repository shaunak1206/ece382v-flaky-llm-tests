import json
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, date
from unittest import mock

import pytest

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
    label: str = 'fixed'


@dataclass
class OnlyInt:
    number: int


# ---------------------------- int_generator ----------------------------


def test_int_generator_returns_int():
    assert isinstance(int_generator(), int)


@pytest.mark.parametrize('length', [0, 1, 3, 6])
def test_int_generator_within_bounds(length):
    for _ in range(50):
        value = int_generator(length)
        assert 0 <= value <= 9 * 10 ** length


def test_int_generator_calls_randint_with_expected_bounds():
    with mock.patch('fistro.generators.randint', return_value=7) as patched:
        assert int_generator(4) == 7
    patched.assert_called_once_with(0, 9 * 10 ** 4)


# ---------------------------- other generators ----------------------------


def test_bool_generator_returns_bool():
    values = {bool_generator() for _ in range(100)}
    assert values <= {True, False}
    assert all(isinstance(v, bool) for v in values)


def test_str_generator_length_and_type():
    result = str_generator(length=12)
    assert isinstance(result, str)
    assert len(result) == 12


def test_str_generator_respects_population():
    result = str_generator(population='ab', length=50)
    assert set(result) <= {'a', 'b'}


def test_str_generator_zero_length():
    assert str_generator(length=0) == ''


def test_datetime_generator_with_fixed_range():
    rand = {
        'min_year': 2020, 'max_year': 2020,
        'min_month': 5, 'max_month': 5,
        'min_day': 10, 'max_day': 10,
        'min_hour': 3, 'max_hour': 3,
        'min_minute': 4, 'max_minute': 4,
        'min_second': 5, 'max_second': 5,
    }
    assert datetime_generator(rand) == datetime(2020, 5, 10, 3, 4, 5)


def test_datetime_generator_default_returns_datetime():
    assert isinstance(datetime_generator(), datetime)


def test_date_generator_with_fixed_range():
    rand = {
        'min_year': 2001, 'max_year': 2001,
        'min_month': 2, 'max_month': 2,
        'min_day': 3, 'max_day': 3,
    }
    result = date_generator(rand)
    assert result == date(2001, 2, 3)
    assert not isinstance(result, datetime)


def test_int_list_generator_length_and_bounds():
    result = int_list_generator(list_length=8, int_length=2)
    assert len(result) == 8
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in result)


def test_int_dict_generator_structure():
    result = int_dict_generator(dict_length=5, int_length=2)
    assert isinstance(result, dict)
    assert 1 <= len(result) <= 5
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in result.values())


def test_default_generators_contents():
    gens = default_generators()
    assert int_generator in gens
    assert str_generator in gens
    assert bool_generator in gens
    assert all(callable(g) for g in gens)
    assert len(gens) == 7


# ---------------------------- generate ----------------------------


def test_generate_returns_dataclass_with_same_name_and_fields():
    result = generate(Simple)
    assert is_dataclass(result)
    assert result.__name__ == 'Simple'
    assert [f.name for f in fields(result)] == ['number', 'text', 'flag']


def test_generate_instance_has_correct_types():
    instance = generate(Simple)()
    assert isinstance(instance.number, int)
    assert isinstance(instance.text, str)
    assert isinstance(instance.flag, bool)


def test_generate_as_dict():
    result = generate(Simple, as_dict=True)
    assert isinstance(result, dict)
    assert set(result) == {'number', 'text', 'flag'}
    assert isinstance(result['number'], int)
    assert isinstance(result['text'], str)
    assert isinstance(result['flag'], bool)


def test_generate_as_json_is_valid_json():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded) == {'number', 'text', 'flag'}


def test_generate_keeps_default_values():
    result = generate(WithDefault, as_dict=True)
    assert result['label'] == 'fixed'
    assert isinstance(result['number'], int)


def test_generate_with_custom_generator():
    def constant_int() -> int:
        return 42

    result = generate(OnlyInt, generators=[constant_int], as_dict=True)
    assert result == {'number': 42}


def test_generate_instances_can_be_overridden():
    instance = generate(Simple)(number=1, text='x', flag=False)
    assert (instance.number, instance.text, instance.flag) == (1, 'x', False)
