import json
from dataclasses import dataclass, is_dataclass, fields
from datetime import datetime, date

import pytest

from fistro import generators
from fistro.config import INT_LENGTH, INT_LIST_LENGTH, STR_LENGTH
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
class Simple:
    number: int
    text: str
    flag: bool


@dataclass
class WithDefault:
    number: int
    fixed: int = 7


# ---------------------------- int_generator ----------------------------

def test_int_generator_returns_int():
    assert isinstance(int_generator(), int)


def test_int_generator_default_range():
    for _ in range(200):
        value = int_generator()
        assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_custom_length_range():
    for _ in range(200):
        assert 0 <= int_generator(2) <= 900


def test_int_generator_length_zero():
    for _ in range(100):
        assert 0 <= int_generator(0) <= 9


def test_int_generator_calls_randint_with_expected_bounds(monkeypatch):
    monkeypatch.setattr(generators, 'randint', lambda a, b: (a, b))
    assert int_generator(3) == (0, 9 * 10 ** 3)


def test_int_generator_default_bounds_use_config(monkeypatch):
    monkeypatch.setattr(generators, 'randint', lambda a, b: (a, b))
    assert int_generator() == (0, 9 * 10 ** INT_LENGTH)


# ---------------------------- other generators ----------------------------

def test_bool_generator_returns_bool():
    assert all(isinstance(bool_generator(), bool) for _ in range(20))


def test_str_generator_length_and_type():
    result = str_generator(length=12)
    assert isinstance(result, str)
    assert len(result) == 12


def test_str_generator_default_length():
    assert len(str_generator()) == STR_LENGTH


def test_str_generator_population_respected():
    result = str_generator(population='ab', length=50)
    assert set(result) <= {'a', 'b'}


def test_datetime_generator_pinned_values():
    rand = {
        'min_year': 2000, 'max_year': 2000,
        'min_month': 5, 'max_month': 5,
        'min_day': 10, 'max_day': 10,
        'min_hour': 3, 'max_hour': 3,
        'min_minute': 4, 'max_minute': 4,
        'min_second': 5, 'max_second': 5,
    }
    assert datetime_generator(rand) == datetime(2000, 5, 10, 3, 4, 5)


def test_date_generator_pinned_values():
    rand = {
        'min_year': 1999, 'max_year': 1999,
        'min_month': 12, 'max_month': 12,
        'min_day': 1, 'max_day': 1,
    }
    result = date_generator(rand)
    assert result == date(1999, 12, 1)
    assert type(result) is date


def test_int_list_generator_length_and_bounds():
    result = int_list_generator(list_length=5, int_length=1)
    assert len(result) == 5
    assert all(isinstance(x, int) and 0 <= x <= 90 for x in result)


def test_int_dict_generator_types():
    result = int_dict_generator(dict_length=3, int_length=1)
    assert 1 <= len(result) <= 3
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in result.items())


def test_default_generators_are_callable_and_complete():
    gens = default_generators()
    assert int_generator in gens
    assert str_generator in gens
    assert all(callable(g) for g in gens)


# ---------------------------- generate ----------------------------

def test_generate_returns_dataclass():
    result = generate(Simple)
    assert is_dataclass(result)
    assert isinstance(result, type)
    assert result.__name__ == 'Simple'
    assert [f.name for f in fields(result)] == ['number', 'text', 'flag']


def test_generate_instance_has_generated_types():
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


def test_generate_as_json():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    loaded = json.loads(result)
    assert set(loaded) == {'number', 'text', 'flag'}
    assert isinstance(loaded['number'], int)
    assert isinstance(loaded['flag'], bool)


def test_generate_preserves_default_values():
    for _ in range(10):
        assert generate(WithDefault)().fixed == 7
    assert generate(WithDefault, as_dict=True)['fixed'] == 7


def test_generate_with_custom_generator():
    def constant_int() -> int:
        return 42

    result = generate(WithDefault, generators=[constant_int], as_dict=True)
    assert result['number'] == 42
    assert result['fixed'] == 7
