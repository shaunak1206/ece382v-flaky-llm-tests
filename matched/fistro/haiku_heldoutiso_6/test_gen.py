import json
import random
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime

import pytest

import fistro.generators as gen
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
class Simple:
    number: int
    text: str
    flag: bool


@dataclass
class WithDefault:
    number: int = 7


# ---------------------------------------------------------------- int_generator

def test_int_generator_returns_int_within_default_range():
    for _ in range(50):
        value = int_generator()
        assert isinstance(value, int)
        assert 0 <= value <= 9 * 10 ** INT_LENGTH


def test_int_generator_passes_expected_bounds_to_randint(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return a

    monkeypatch.setattr(gen, 'randint', fake_randint)
    assert int_generator(3) == 0
    assert calls == [(0, 9000)]


def test_int_generator_zero_length_gives_single_digit_range(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return b

    monkeypatch.setattr(gen, 'randint', fake_randint)
    assert int_generator(0) == 9
    assert calls == [(0, 9)]


def test_int_generator_respects_length_argument():
    for _ in range(100):
        value = int_generator(1)
        assert 0 <= value <= 90


# ---------------------------------------------------------------- bool / str

def test_bool_generator_returns_bool():
    for _ in range(20):
        assert isinstance(bool_generator(), bool)


def test_bool_generator_chooses_from_true_and_false(monkeypatch):
    seen = []

    def fake_choice(seq):
        seen.append(list(seq))
        return seq[0]

    monkeypatch.setattr(gen, 'choice', fake_choice)
    assert bool_generator() is True
    assert seen == [[True, False]]


def test_str_generator_default_length_and_population():
    value = str_generator()
    assert isinstance(value, str)
    assert len(value) == STR_LENGTH
    assert set(value) <= set(string.printable)


def test_str_generator_custom_population_and_length():
    value = str_generator('ab', 5)
    assert len(value) == 5
    assert set(value) <= {'a', 'b'}


def test_str_generator_zero_length_returns_empty_string():
    assert str_generator('xyz', 0) == ''


# ---------------------------------------------------------------- datetime_generator

def test_datetime_generator_with_fixed_bounds_is_deterministic():
    rand_date = {
        'min_year': 2000, 'max_year': 2000,
        'min_month': 1, 'max_month': 1,
        'min_day': 2, 'max_day': 2,
        'min_hour': 3, 'max_hour': 3,
        'min_minute': 4, 'max_minute': 4,
        'min_second': 5, 'max_second': 5,
    }
    result = datetime_generator(rand_date)
    assert isinstance(result, datetime)
    assert result == datetime(2000, 1, 2, 3, 4, 5)


def test_datetime_generator_none_argument_returns_datetime():
    result = datetime_generator(None)
    assert isinstance(result, datetime)


def test_datetime_generator_values_within_random_bounds():
    rand_date = {
        'min_year': 1990, 'max_year': 1995,
        'min_month': 3, 'max_month': 9,
        'min_day': 1, 'max_day': 28,
        'min_hour': 0, 'max_hour': 23,
        'min_minute': 0, 'max_minute': 59,
        'min_second': 0, 'max_second': 59,
    }
    random.seed(1234)
    for _ in range(30):
        result = datetime_generator(rand_date)
        assert 1990 <= result.year <= 1995
        assert 3 <= result.month <= 9
        assert 1 <= result.day <= 28
        assert 0 <= result.hour <= 23
        assert 0 <= result.minute <= 59
        assert 0 <= result.second <= 59


# ---------------------------------------------------------------- date_generator

def test_date_generator_returns_plain_date_not_datetime():
    rand_date = {
        'min_year': 1999, 'max_year': 1999,
        'min_month': 12, 'max_month': 12,
        'min_day': 31, 'max_day': 31,
    }
    result = date_generator(rand_date)
    assert type(result) is date
    assert result == date(1999, 12, 31)


def test_date_generator_values_within_random_bounds():
    rand_date = {
        'min_year': 2010, 'max_year': 2012,
        'min_month': 1, 'max_month': 6,
        'min_day': 1, 'max_day': 15,
    }
    random.seed(99)
    for _ in range(30):
        result = date_generator(rand_date)
        assert isinstance(result, date)
        assert 2010 <= result.year <= 2012
        assert 1 <= result.month <= 6
        assert 1 <= result.day <= 15


# ---------------------------------------------------------------- list / dict generators

def test_int_list_generator_length_and_element_types():
    result = int_list_generator(list_length=4, int_length=2)
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(v, int) and 0 <= v <= 900 for v in result)


def test_int_list_generator_default_length():
    assert len(int_list_generator()) == INT_LIST_LENGTH


def test_int_dict_generator_string_keys_and_int_values(monkeypatch):
    counter = iter(range(1000))
    monkeypatch.setattr(gen, 'str_generator', lambda: f'key{next(counter)}')
    result = int_dict_generator(dict_length=3, int_length=1)
    assert len(result) == 3
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) and 0 <= v <= 90 for v in result.values())


def test_default_generators_contains_expected_callables():
    generators = default_generators()
    assert isinstance(generators, list)
    assert len(generators) == 7
    assert all(callable(g) for g in generators)
    for expected in (int_generator, bool_generator, str_generator,
                     datetime_generator, date_generator,
                     int_list_generator, int_dict_generator):
        assert expected in generators


# ---------------------------------------------------------------- generate

def test_generate_returns_dataclass_with_same_name_and_fields():
    result_cls = generate(Simple)
    assert is_dataclass(result_cls)
    assert result_cls.__name__ == 'Simple'
    assert [f.name for f in fields(result_cls)] == ['number', 'text', 'flag']


def test_generate_instance_has_values_of_declared_types():
    result_cls = generate(Simple)
    instance = result_cls()
    assert isinstance(instance.number, int)
    assert isinstance(instance.text, str)
    assert isinstance(instance.flag, bool)


def test_generate_as_dict_returns_dict_with_generated_values():
    result = generate(Simple, as_dict=True)
    assert isinstance(result, dict)
    assert set(result) == {'number', 'text', 'flag'}
    assert isinstance(result['number'], int)
    assert isinstance(result['text'], str)
    assert isinstance(result['flag'], bool)


def test_generate_as_json_returns_valid_json_string():
    result = generate(Simple, as_json=True)
    assert isinstance(result, str)
    data = json.loads(result)
    assert set(data) == {'number', 'text', 'flag'}
    assert isinstance(data['number'], int)
    assert isinstance(data['flag'], bool)


def test_generate_as_dict_takes_precedence_over_as_json():
    result = generate(Simple, as_dict=True, as_json=True)
    assert isinstance(result, dict)


def test_generate_preserves_declared_default_values():
    instance = generate(WithDefault)()
    assert instance.number == 7


def test_generate_uses_custom_generator_for_matching_type():
    def my_int() -> int:
        return 42

    result_cls = generate(Simple, generators=[my_int])
    instance = result_cls()
    assert instance.number == 42
