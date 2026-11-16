import json
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime

import pytest

from fistro import generators as gen
from fistro.config import INT_LENGTH, INT_LIST_LENGTH, STR_LENGTH
from fistro.fistro import generate


@dataclass
class Person:
    name: str
    age: int


@dataclass
class Counter:
    count: int


@dataclass
class WithDefault:
    level: int = 5


# ---------------------------------------------------------------------------
# int_generator
# ---------------------------------------------------------------------------

def test_int_generator_returns_int():
    assert isinstance(gen.int_generator(), int)


def test_int_generator_default_length_passes_expected_bounds(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return a

    monkeypatch.setattr(gen, 'randint', fake_randint)
    gen.int_generator()
    assert calls == [(0, 9 * 10 ** INT_LENGTH)]


def test_int_generator_custom_length_passes_expected_bounds(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return a

    monkeypatch.setattr(gen, 'randint', fake_randint)
    gen.int_generator(3)
    assert calls == [(0, 9000)]


def test_int_generator_returns_value_from_randint(monkeypatch):
    monkeypatch.setattr(gen, 'randint', lambda a, b: 1234)
    assert gen.int_generator() == 1234


def test_int_generator_length_one_values_in_range():
    for _ in range(200):
        value = gen.int_generator(1)
        assert 0 <= value <= 90


def test_int_generator_length_zero_values_in_range():
    for _ in range(200):
        value = gen.int_generator(0)
        assert 0 <= value <= 9


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_returns_dataclass_with_same_name():
    result = generate(Person)
    assert is_dataclass(result)
    assert result.__name__ == 'Person'


def test_generate_preserves_field_names():
    result = generate(Person)
    assert [f.name for f in fields(result)] == ['name', 'age']


def test_generate_default_instance_has_attributes():
    instance = generate(Person)()
    assert hasattr(instance, 'name')
    assert hasattr(instance, 'age')


def test_generate_as_dict_returns_dict_with_field_keys():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'name', 'age'}


def test_generate_as_dict_values_match_field_types():
    result = generate(Person, as_dict=True)
    assert isinstance(result['name'], str)
    assert isinstance(result['age'], int)


def test_generate_as_json_returns_valid_json_string():
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert set(parsed.keys()) == {'name', 'age'}


def test_generate_keeps_default_values():
    result = generate(WithDefault, as_dict=True)
    assert result == {'level': 5}


def test_generate_custom_generator_is_used():
    def always_seven() -> int:
        return 7

    result = generate(Counter, generators=[always_seven], as_dict=True)
    assert result == {'count': 7}


def test_generate_returns_new_class_each_call():
    first = generate(Person)
    second = generate(Person)
    assert first is not second


# ---------------------------------------------------------------------------
# generators.py: bool, str, datetime, date, lists, dicts, defaults
# ---------------------------------------------------------------------------

def test_bool_generator_returns_bool():
    for _ in range(50):
        assert isinstance(gen.bool_generator(), bool)


def test_str_generator_default_length_and_population():
    value = gen.str_generator()
    assert len(value) == STR_LENGTH
    assert all(ch in string.printable for ch in value)


def test_str_generator_custom_population_and_length():
    value = gen.str_generator(population='ab', length=5)
    assert len(value) == 5
    assert set(value) <= {'a', 'b'}


def test_datetime_generator_fixed_range_returns_exact_datetime():
    rand_date = {
        'min_year': 2000, 'max_year': 2000,
        'min_month': 1, 'max_month': 1,
        'min_day': 2, 'max_day': 2,
        'min_hour': 3, 'max_hour': 3,
        'min_minute': 4, 'max_minute': 4,
        'min_second': 5, 'max_second': 5,
    }
    assert gen.datetime_generator(rand_date) == datetime(2000, 1, 2, 3, 4, 5)


def test_datetime_generator_default_returns_datetime():
    assert isinstance(gen.datetime_generator(), datetime)


def test_date_generator_fixed_range_returns_exact_date():
    rand_date = {
        'min_year': 1999, 'max_year': 1999,
        'min_month': 12, 'max_month': 12,
        'min_day': 31, 'max_day': 31,
    }
    assert gen.date_generator(rand_date) == date(1999, 12, 31)


def test_date_generator_returns_date_not_datetime():
    result = gen.date_generator()
    assert isinstance(result, date)
    assert not isinstance(result, datetime)


def test_int_list_generator_length_zero_returns_empty_list():
    assert gen.int_list_generator(0) == []


def test_int_list_generator_returns_requested_length_of_ints():
    result = gen.int_list_generator(5, 2)
    assert len(result) == 5
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in result)


def test_int_dict_generator_returns_str_keys_and_int_values(monkeypatch):
    counter = iter(range(1000))
    monkeypatch.setattr(gen, 'str_generator', lambda: f'key{next(counter)}')
    result = gen.int_dict_generator(3, 1)
    assert len(result) == 3
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) for v in result.values())


def test_default_generators_returns_all_generator_callables():
    result = gen.default_generators()
    assert len(result) == 7
    assert all(callable(g) for g in result)
    assert gen.int_generator in result
