import json
import string
from dataclasses import dataclass, fields
from datetime import date, datetime

import pytest

from fistro import generators
from fistro.config import INT_LENGTH, INT_LIST_LENGTH, STR_LENGTH
from fistro.fistro import generate
from fistro.generators import (
    bool_generator,
    date_generator,
    datetime_generator,
    default_generators,
    int_dict_generator,
    int_generator,
    int_list_generator,
    str_generator,
)


@dataclass
class Person:
    name: str
    age: int
    active: bool


@dataclass
class WithDefault:
    count: int = 5


@dataclass
class Number:
    value: int


def const_int() -> int:
    return 42


def _recording_randint(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return a

    monkeypatch.setattr(generators, 'randint', fake_randint)
    return calls


# ---------------------------------------------------------------- generate

def test_generate_returns_dataclass_with_same_name():
    result = generate(Person)
    assert result.__name__ == 'Person'
    assert hasattr(result, '__dataclass_fields__')


def test_generate_returns_same_field_names_and_types():
    result = generate(Person)
    names = [f.name for f in fields(result)]
    types = {f.name: f.type for f in fields(result)}
    assert names == ['name', 'age', 'active']
    assert types == {'name': str, 'age': int, 'active': bool}


def test_generate_instance_values_match_field_types():
    instance = generate(Person)()
    assert isinstance(instance.name, str)
    assert isinstance(instance.age, int)
    assert isinstance(instance.active, bool)


def test_generate_as_dict_returns_dict_with_field_keys():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'name', 'age', 'active'}
    assert isinstance(result['name'], str)
    assert isinstance(result['age'], int)
    assert isinstance(result['active'], bool)


def test_generate_as_json_returns_valid_json_string():
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    data = json.loads(result)
    assert set(data.keys()) == {'name', 'age', 'active'}
    assert isinstance(data['age'], int)


def test_generate_as_dict_takes_precedence_over_as_json():
    result = generate(Person, as_dict=True, as_json=True)
    assert isinstance(result, dict)


def test_generate_preserves_default_values():
    cls = generate(WithDefault)
    assert cls().count == 5


def test_generate_uses_custom_generators():
    cls = generate(Number, generators=[const_int])
    assert cls().value == 42


# ---------------------------------------------------------------- int_generator

def test_int_generator_default_length_bounds_randint(monkeypatch):
    calls = _recording_randint(monkeypatch)
    int_generator()
    assert calls == [(0, 9 * 10 ** INT_LENGTH)]


def test_int_generator_custom_length_bounds_randint(monkeypatch):
    calls = _recording_randint(monkeypatch)
    int_generator(3)
    assert calls == [(0, 9 * 10 ** 3)]


def test_int_generator_returns_int_within_bounds():
    for _ in range(100):
        value = int_generator(2)
        assert isinstance(value, int)
        assert 0 <= value <= 900


def test_int_generator_length_zero_gives_single_digit_range(monkeypatch):
    calls = _recording_randint(monkeypatch)
    int_generator(0)
    assert calls == [(0, 9)]


# ---------------------------------------------------------------- other generators

def test_str_generator_default_length_and_population():
    value = str_generator()
    assert isinstance(value, str)
    assert len(value) == STR_LENGTH
    assert all(ch in string.printable for ch in value)


def test_str_generator_custom_population_and_length():
    value = str_generator(population='ab', length=10)
    assert len(value) == 10
    assert set(value) <= {'a', 'b'}


def test_bool_generator_returns_bool_and_both_values_occur():
    values = {bool_generator() for _ in range(200)}
    assert all(isinstance(v, bool) for v in values)
    assert values == {True, False}


def test_int_list_generator_length_and_element_types():
    value = int_list_generator(list_length=5, int_length=2)
    assert isinstance(value, list)
    assert len(value) == 5
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in value)


def test_int_list_generator_default_length():
    assert len(int_list_generator()) == INT_LIST_LENGTH


def test_int_dict_generator_uses_str_keys_and_int_values(monkeypatch):
    counter = iter(range(1000))
    monkeypatch.setattr(
        generators, 'str_generator', lambda: f'key{next(counter)}'
    )
    value = int_dict_generator(dict_length=4, int_length=1)
    assert len(value) == 4
    assert all(isinstance(k, str) for k in value)
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in value.values())


def test_datetime_generator_fixed_range_is_deterministic():
    rand_date = {
        'min_year': 2020, 'max_year': 2020,
        'min_month': 5, 'max_month': 5,
        'min_day': 17, 'max_day': 17,
        'min_hour': 8, 'max_hour': 8,
        'min_minute': 30, 'max_minute': 30,
        'min_second': 15, 'max_second': 15,
    }
    result = datetime_generator(rand_date)
    assert isinstance(result, datetime)
    assert result == datetime(2020, 5, 17, 8, 30, 15)


def test_datetime_generator_respects_bounds():
    rand_date = {
        'min_year': 1999, 'max_year': 2001,
        'min_month': 1, 'max_month': 12,
        'min_day': 1, 'max_day': 28,
        'min_hour': 0, 'max_hour': 23,
        'min_minute': 0, 'max_minute': 59,
        'min_second': 0, 'max_second': 59,
    }
    for _ in range(50):
        result = datetime_generator(rand_date)
        assert isinstance(result, datetime)
        assert datetime(1999, 1, 1) <= result <= datetime(2001, 12, 31, 23, 59, 59)


def test_date_generator_fixed_range_returns_date_not_datetime():
    rand_date = {
        'min_year': 1990, 'max_year': 1990,
        'min_month': 2, 'max_month': 2,
        'min_day': 3, 'max_day': 3,
    }
    result = date_generator(rand_date)
    assert isinstance(result, date)
    assert not isinstance(result, datetime)
    assert result == date(1990, 2, 3)


def test_date_generator_respects_bounds():
    rand_date = {
        'min_year': 2010, 'max_year': 2012,
        'min_month': 3, 'max_month': 4,
        'min_day': 1, 'max_day': 28,
    }
    for _ in range(50):
        result = date_generator(rand_date)
        assert date(2010, 3, 1) <= result <= date(2012, 4, 28)


def test_default_generators_contains_all_generators():
    gens = default_generators()
    assert isinstance(gens, list)
    assert len(gens) == 7
    assert all(callable(g) for g in gens)
    for g in (
        int_generator, bool_generator, str_generator, datetime_generator,
        date_generator, int_list_generator, int_dict_generator,
    ):
        assert g in gens
