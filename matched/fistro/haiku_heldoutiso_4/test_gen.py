import dataclasses
import itertools
import json
from dataclasses import dataclass
from datetime import date, datetime

import pytest

from fistro import generators as gen
from fistro.config import INT_LENGTH, INT_LIST_LENGTH
from fistro.fistro import generate


@dataclass
class Person:
    age: int
    name: str


@dataclass
class Pet:
    name: str = 'rex'


@dataclass
class OnlyInt:
    value: int


def fixed_int_generator() -> int:
    return 42


# ---------- int_generator ----------

def test_int_generator_returns_int():
    assert isinstance(gen.int_generator(), int)


def test_int_generator_default_length_bounds():
    upper = 9 * 10 ** INT_LENGTH
    for _ in range(50):
        value = gen.int_generator()
        assert 0 <= value <= upper


def test_int_generator_zero_length_is_single_digit():
    for _ in range(100):
        value = gen.int_generator(0)
        assert 0 <= value <= 9


def test_int_generator_passes_length_to_randint(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return low

    monkeypatch.setattr(gen, 'randint', fake_randint)
    assert gen.int_generator(3) == 0
    assert calls == [(0, 9 * 10 ** 3)]


# ---------- other scalar generators ----------

def test_bool_generator_returns_bool():
    assert isinstance(gen.bool_generator(), bool)


def test_bool_generator_uses_choice(monkeypatch):
    monkeypatch.setattr(gen, 'choice', lambda options: True)
    assert gen.bool_generator() is True


def test_str_generator_default_length():
    value = gen.str_generator()
    assert isinstance(value, str)
    assert len(value) == gen.STR_LENGTH


def test_str_generator_respects_population_and_length():
    value = gen.str_generator(population='ab', length=5)
    assert len(value) == 5
    assert set(value) <= {'a', 'b'}


# ---------- datetime / date generators ----------

def test_datetime_generator_returns_datetime():
    assert isinstance(gen.datetime_generator(), datetime)


def test_datetime_generator_fixed_bounds():
    rand_date = {
        'min_year': 2000, 'max_year': 2000,
        'min_month': 1, 'max_month': 1,
        'min_day': 2, 'max_day': 2,
        'min_hour': 3, 'max_hour': 3,
        'min_minute': 4, 'max_minute': 4,
        'min_second': 5, 'max_second': 5,
    }
    assert gen.datetime_generator(rand_date) == datetime(2000, 1, 2, 3, 4, 5)


def test_date_generator_returns_plain_date():
    value = gen.date_generator()
    assert type(value) is date


def test_date_generator_fixed_bounds():
    rand_date = {
        'min_year': 2001, 'max_year': 2001,
        'min_month': 2, 'max_month': 2,
        'min_day': 3, 'max_day': 3,
    }
    assert gen.date_generator(rand_date) == date(2001, 2, 3)


# ---------- collection generators ----------

def test_int_list_generator_length_and_types():
    values = gen.int_list_generator(list_length=4, int_length=2)
    assert len(values) == 4
    assert all(isinstance(v, int) and 0 <= v <= 9 * 10 ** 2 for v in values)


def test_int_list_generator_zero_length():
    assert gen.int_list_generator(list_length=0) == []


def test_int_dict_generator_uses_str_keys_and_int_values(monkeypatch):
    counter = itertools.count()
    monkeypatch.setattr(gen, 'str_generator', lambda: f'key{next(counter)}')
    result = gen.int_dict_generator(dict_length=3, int_length=1)
    assert list(result.keys()) == ['key0', 'key1', 'key2']
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


def test_default_generators_contents():
    funcs = gen.default_generators()
    assert isinstance(funcs, list)
    assert len(funcs) == 7
    assert gen.int_generator in funcs
    assert gen.bool_generator in funcs
    assert gen.str_generator in funcs
    assert gen.datetime_generator in funcs
    assert gen.date_generator in funcs
    assert gen.int_list_generator in funcs
    assert gen.int_dict_generator in funcs


# ---------- generate ----------

def test_generate_returns_dataclass_with_same_name():
    result = generate(Person)
    assert dataclasses.is_dataclass(result)
    assert result.__name__ == 'Person'


def test_generate_as_dict_returns_populated_dict():
    result = generate(Person, as_dict=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'age', 'name'}
    assert isinstance(result['age'], int)
    assert isinstance(result['name'], str)


def test_generate_as_json_returns_valid_json():
    result = generate(Person, as_json=True)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert set(parsed.keys()) == {'age', 'name'}
    assert isinstance(parsed['age'], int)
    assert isinstance(parsed['name'], str)


def test_generate_preserves_default_values():
    result = generate(Pet, as_dict=True)
    assert result == {'name': 'rex'}


def test_generate_default_instance_has_default_value():
    cls = generate(Pet)
    assert cls().name == 'rex'


def test_generate_uses_custom_generators():
    result = generate(OnlyInt, generators=[fixed_int_generator], as_dict=True)
    assert result == {'value': 42}


def test_generate_without_dict_or_json_returns_instantiable_class():
    cls = generate(OnlyInt)
    instance = cls()
    assert isinstance(instance.value, int)
