import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime

import pytest

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


@dataclass
class Person:
    name: str
    age: int
    active: bool


@dataclass
class WithDefault:
    name: str
    age: int = 30


@dataclass
class OnlyInt:
    value: int


def constant_int() -> int:
    return 42


# ---------------------------- int_generator ----------------------------


def test_int_generator_returns_int():
    assert isinstance(int_generator(), int)


@pytest.mark.parametrize('length', [0, 1, 2, 5])
def test_int_generator_within_bounds(length):
    for _ in range(50):
        value = int_generator(length)
        assert 0 <= value <= 9 * 10 ** length


def test_int_generator_length_zero_is_single_digit():
    for _ in range(50):
        assert 0 <= int_generator(0) <= 9


def test_int_generator_produces_varied_values():
    values = {int_generator(6) for _ in range(30)}
    assert len(values) > 1


# ---------------------------- other generators ----------------------------


def test_bool_generator_returns_bool():
    values = {bool_generator() for _ in range(100)}
    assert values <= {True, False}
    assert all(isinstance(v, bool) for v in values)


def test_str_generator_length_and_population():
    result = str_generator(population='abc', length=20)
    assert isinstance(result, str)
    assert len(result) == 20
    assert set(result) <= set('abc')


def test_str_generator_zero_length():
    assert str_generator(length=0) == ''


def test_datetime_generator_fixed_bounds():
    bounds = {
        'min_year': 2000, 'max_year': 2000,
        'min_month': 5, 'max_month': 5,
        'min_day': 10, 'max_day': 10,
        'min_hour': 7, 'max_hour': 7,
        'min_minute': 30, 'max_minute': 30,
        'min_second': 15, 'max_second': 15,
    }
    assert datetime_generator(bounds) == datetime(2000, 5, 10, 7, 30, 15)


def test_date_generator_fixed_bounds():
    bounds = {
        'min_year': 1999, 'max_year': 1999,
        'min_month': 12, 'max_month': 12,
        'min_day': 25, 'max_day': 25,
    }
    result = date_generator(bounds)
    assert result == date(1999, 12, 25)
    assert not isinstance(result, datetime)


def test_int_list_generator_length_and_range():
    result = int_list_generator(list_length=7, int_length=2)
    assert len(result) == 7
    assert all(isinstance(i, int) and 0 <= i <= 900 for i in result)


def test_int_dict_generator_structure():
    result = int_dict_generator(dict_length=4, int_length=1)
    assert isinstance(result, dict)
    assert 1 <= len(result) <= 4
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


def test_default_generators_are_callables():
    generators = default_generators()
    assert int_generator in generators
    assert str_generator in generators
    assert all(callable(g) for g in generators)


# ---------------------------- generate ----------------------------


def test_generate_returns_dataclass_with_same_name_and_fields():
    result = generate(Person)
    assert is_dataclass(result)
    assert result.__name__ == 'Person'
    assert [f.name for f in fields(result)] == ['name', 'age', 'active']


def test_generate_instance_has_correct_types():
    instance = generate(Person)()
    assert isinstance(instance.name, str)
    assert isinstance(instance.age, int)
    assert isinstance(instance.active, bool)


def test_generate_as_dict():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'name', 'age', 'active'}
    assert isinstance(result['name'], str)
    assert isinstance(result['age'], int)
    assert isinstance(result['active'], bool)


def test_generate_as_json_is_parseable():
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert set(parsed.keys()) == {'name', 'age', 'active'}


def test_generate_keeps_explicit_defaults():
    result = generate(WithDefault, as_dict=True)
    assert result['age'] == 30
    assert isinstance(result['name'], str)

    instance = generate(WithDefault)()
    assert instance.age == 30


def test_generate_with_custom_generator():
    result = generate(OnlyInt, generators=[constant_int], as_dict=True)
    assert result == {'value': 42}
