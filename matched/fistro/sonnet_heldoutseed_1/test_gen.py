import dataclasses
import json
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, date

import pytest

from fistro.config import INT_LENGTH, STR_LENGTH, INT_LIST_LENGTH
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
    age: int = 7


@dataclass
class OnlyInt:
    value: int


# ---------------------------------------------------------------- int_generator

def test_int_generator_returns_int():
    assert isinstance(int_generator(), int)


def test_int_generator_default_range():
    for _ in range(50):
        value = int_generator()
        assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_length_zero_range():
    for _ in range(50):
        assert 0 <= int_generator(0) <= 9


def test_int_generator_custom_length_range():
    for _ in range(50):
        assert 0 <= int_generator(3) <= 9000


def test_int_generator_produces_varied_values():
    values = {int_generator(6) for _ in range(30)}
    assert len(values) > 1


# ------------------------------------------------------------- other generators

def test_bool_generator_returns_bool():
    for _ in range(20):
        assert isinstance(bool_generator(), bool)


def test_str_generator_default_length():
    result = str_generator()
    assert isinstance(result, str)
    assert len(result) == STR_LENGTH


def test_str_generator_custom_population_and_length():
    result = str_generator(population='ab', length=25)
    assert len(result) == 25
    assert set(result) <= {'a', 'b'}


def test_str_generator_default_population_is_printable():
    result = str_generator(length=100)
    assert all(c in string.printable for c in result)


def test_datetime_generator_respects_bounds():
    bounds = {
        'min_year': 2020, 'max_year': 2020,
        'min_month': 5, 'max_month': 5,
        'min_day': 15, 'max_day': 15,
        'min_hour': 3, 'max_hour': 3,
        'min_minute': 30, 'max_minute': 30,
        'min_second': 10, 'max_second': 10,
    }
    assert datetime_generator(bounds) == datetime(2020, 5, 15, 3, 30, 10)


def test_date_generator_respects_bounds():
    bounds = {
        'min_year': 1999, 'max_year': 1999,
        'min_month': 12, 'max_month': 12,
        'min_day': 1, 'max_day': 1,
    }
    result = date_generator(bounds)
    assert result == date(1999, 12, 1)
    assert type(result) is date


def test_datetime_generator_default_returns_datetime():
    assert isinstance(datetime_generator(), datetime)


def test_int_list_generator_length_and_values():
    result = int_list_generator(list_length=5, int_length=2)
    assert len(result) == 5
    assert all(isinstance(x, int) and 0 <= x <= 900 for x in result)


def test_int_list_generator_default_length():
    assert len(int_list_generator()) == INT_LIST_LENGTH


def test_int_dict_generator_structure():
    result = int_dict_generator(dict_length=3, int_length=1)
    assert isinstance(result, dict)
    assert 1 <= len(result) <= 3
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


def test_default_generators_contents():
    gens = default_generators()
    assert len(gens) == 7
    assert int_generator in gens
    assert str_generator in gens
    assert all(callable(g) for g in gens)


# --------------------------------------------------------------------- generate

def test_generate_returns_dataclass_with_same_name_and_fields():
    result = generate(Person)
    assert is_dataclass(result)
    assert result.__name__ == 'Person'
    assert [f.name for f in fields(result)] == ['name', 'age', 'active']


def test_generate_instance_has_random_typed_values():
    instance = generate(Person)()
    assert isinstance(instance.name, str)
    assert isinstance(instance.age, int) and not isinstance(instance.age, bool)
    assert isinstance(instance.active, bool)


def test_generate_as_dict():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result) == {'name', 'age', 'active'}
    assert isinstance(result['name'], str)
    assert isinstance(result['age'], int)
    assert isinstance(result['active'], bool)


def test_generate_as_json_is_valid_json():
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded) == {'name', 'age', 'active'}


def test_generate_keeps_default_values():
    instance = generate(WithDefault)()
    assert instance.age == 7
    assert isinstance(instance.name, str)
    assert generate(WithDefault, as_dict=True)['age'] == 7


def test_generate_with_custom_generator():
    def constant_int() -> int:
        return 42

    result = generate(OnlyInt, generators=[constant_int], as_dict=True)
    assert result == {'value': 42}


def test_generate_creates_different_instances():
    cls = generate(Person)
    names = {cls().name for _ in range(10)}
    assert len(names) > 1
