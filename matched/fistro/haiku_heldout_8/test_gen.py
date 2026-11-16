import itertools
import json
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from typing import List

import pytest

import fistro.generators as generators
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
class WithDefaults:
    label: str = 'hello'
    count: int = 5


@dataclass
class IntOnly:
    value: int


def fixed_int() -> int:
    return 7


# ---------- generate ----------

def test_generate_returns_dataclass_by_default():
    result = generate(Person)
    assert is_dataclass(result)
    assert result.__name__ == 'Person'


def test_generate_preserves_field_names_and_types():
    result = generate(Person)
    assert [f.name for f in fields(result)] == ['name', 'age', 'active']
    assert [f.type for f in fields(result)] == [str, int, bool]


def test_generate_as_dict_returns_dict_with_generated_values():
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


def test_generate_keeps_default_values():
    result = generate(WithDefaults, as_dict=True)
    assert result == {'label': 'hello', 'count': 5}


def test_generate_uses_custom_generators():
    result = generate(IntOnly, generators=[fixed_int], as_dict=True)
    assert result == {'value': 7}


def test_generate_empty_generators_falls_back_to_defaults():
    result = generate(IntOnly, generators=[], as_dict=True)
    assert isinstance(result['value'], int)


def test_generate_produces_new_dataclass_type_each_call():
    first = generate(IntOnly)
    second = generate(IntOnly)
    assert first is not second
    assert first.__name__ == second.__name__ == 'IntOnly'


# ---------- int_generator ----------

def test_int_generator_default_length_bounds(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return high

    monkeypatch.setattr(generators, 'randint', fake_randint)
    assert int_generator() == 9 * 10 ** INT_LENGTH
    assert calls == [(0, 9 * 10 ** INT_LENGTH)]


def test_int_generator_explicit_length_bounds():
    for _ in range(100):
        value = int_generator(2)
        assert 0 <= value <= 900


def test_int_generator_zero_length_bounds(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return low

    monkeypatch.setattr(generators, 'randint', fake_randint)
    assert int_generator(0) == 0
    assert calls == [(0, 9)]


def test_int_generator_returns_int():
    assert isinstance(int_generator(), int)


# ---------- bool_generator ----------

def test_bool_generator_returns_both_values():
    seen = {bool_generator() for _ in range(200)}
    assert seen == {True, False}


# ---------- str_generator ----------

def test_str_generator_default_length():
    value = str_generator()
    assert isinstance(value, str)
    assert len(value) == STR_LENGTH
    assert all(ch in string.printable for ch in value)


def test_str_generator_custom_population_and_length():
    value = str_generator(population='ab', length=5)
    assert len(value) == 5
    assert set(value) <= {'a', 'b'}


# ---------- datetime_generator ----------

def test_datetime_generator_fixed_bounds_returns_exact_value():
    fixed = {
        'min_year': 2021, 'max_year': 2021,
        'min_month': 3, 'max_month': 3,
        'min_day': 4, 'max_day': 4,
        'min_hour': 5, 'max_hour': 5,
        'min_minute': 6, 'max_minute': 6,
        'min_second': 7, 'max_second': 7,
    }
    result = datetime_generator(fixed)
    assert isinstance(result, datetime)
    assert result == datetime(2021, 3, 4, 5, 6, 7)


def test_datetime_generator_components_within_ranges():
    bounds = {
        'min_year': 2000, 'max_year': 2010,
        'min_month': 1, 'max_month': 12,
        'min_day': 1, 'max_day': 28,
        'min_hour': 0, 'max_hour': 23,
        'min_minute': 0, 'max_minute': 59,
        'min_second': 0, 'max_second': 59,
    }
    for _ in range(50):
        result = datetime_generator(bounds)
        assert 2000 <= result.year <= 2010
        assert 1 <= result.month <= 12
        assert 1 <= result.day <= 28
        assert 0 <= result.hour <= 23
        assert 0 <= result.minute <= 59
        assert 0 <= result.second <= 59


# ---------- date_generator ----------

def test_date_generator_fixed_bounds_returns_exact_date():
    fixed = {
        'min_year': 2020, 'max_year': 2020,
        'min_month': 2, 'max_month': 2,
        'min_day': 29, 'max_day': 29,
    }
    result = date_generator(fixed)
    assert type(result) is date
    assert result == date(2020, 2, 29)


def test_date_generator_invalid_combination_raises():
    invalid = {
        'min_year': 2021, 'max_year': 2021,
        'min_month': 2, 'max_month': 2,
        'min_day': 30, 'max_day': 30,
    }
    with pytest.raises(ValueError):
        date_generator(invalid)


# ---------- int_list_generator / int_dict_generator ----------

def test_int_list_generator_length_and_bounds():
    result = int_list_generator(list_length=4, int_length=1)
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(x, int) and 0 <= x <= 90 for x in result)


def test_int_list_generator_default_length():
    assert len(int_list_generator()) == INT_LIST_LENGTH


def test_int_dict_generator_uses_str_keys_and_int_values(monkeypatch):
    counter = itertools.count()
    monkeypatch.setattr(
        generators, 'str_generator', lambda: f'key{next(counter)}'
    )
    result = int_dict_generator(dict_length=3, int_length=1)
    assert list(result.keys()) == ['key0', 'key1', 'key2']
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


# ---------- default_generators ----------

def test_default_generators_contains_all_generators():
    gens: List = default_generators()
    assert len(gens) == 7
    for func in (
        int_generator,
        bool_generator,
        str_generator,
        datetime_generator,
        date_generator,
        int_list_generator,
        int_dict_generator,
    ):
        assert func in gens
